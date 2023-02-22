"""
Attention modules:
"""

import torch
import torch.nn as nn

from models.model_utils import init_xavier_


class SlotAttention(nn.Module):
    """
    Implementation of the SlotAttention module from:
      --> Locatello, Francesco, et al. "Object-centric learning with slot attention." NeurIPS 2020

    Args:
    -----
    dim_feats: integer
        Dimensionality of the input embeddings
    dim_slots: integer
        Dimensionality of the object slots
    Num_slots: integer
        Number of slots competing for representing the image
    num_iters_first: integer
        Nubmer of recurrent iterations to refine the slots for the first frame in the sequence.
    num_iters: integer
        Nubmer of recurrent iterations to refine the slots from the second frame onwards.
    mlp_hidden_size: integer
        Hidden dimensionality of the mlp,
    epsilon: float
        Small value used to stabilize divisiona and softmax
    """

    def __init__(self, dim_feats, dim_slots, num_slots, num_iters_first=2, num_iters=2,
                 mlp_hidden=128, epsilon=1e-8):
        """
        Module Initializer
        """
        super().__init__()
        self.dim_slots = dim_slots
        self.num_iters_first = num_iters_first
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.epsilon = epsilon
        self.scale = dim_feats ** -0.5

        # normalization layers
        self.norm_input = nn.LayerNorm(dim_feats, eps=0.001)
        self.norm_slot = nn.LayerNorm(dim_slots, eps=0.001)
        self.norm_mlp = nn.LayerNorm(dim_slots, eps=0.001)

        # attention embedders
        self.to_q = nn.Linear(dim_slots, dim_slots)
        self.to_k = nn.Linear(dim_feats, dim_slots)
        self.to_v = nn.Linear(dim_feats, dim_slots)

        # Slot update functions.
        self.gru = nn.GRUCell(dim_slots, dim_slots)
        self.mlp = nn.Sequential(
            nn.Linear(dim_slots, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, dim_slots),
        )
        return

    def forward(self, inputs, slots, step=0, **kwargs):
        """
        Forward pass as depicted in Algorithm 1 from paper

        Args:
        -----
        inputs: torch Tensor
            input feature vectors extracted by the encoder.
            Shape is (Batch, Num locations, Dimensionality)

        Returns:
        --------
        slots: torch Tensor
            Slot assignment for each of the input vectors
            Shape is (Batch, Num Slots, Slot Dimensionality)
        """
        B, N, D = inputs.shape
        self.attention_masks = None

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # iterative refinement of the slot representation
        num_iters = self.num_iters_first if step == 0 else self.num_iters
        for _ in range(num_iters):
            slots_prev = slots
            slots = self.norm_slot(slots)
            q = self.to_q(slots)

            # q ~ (B, N_Slots, Slot_dim)
            # k, v ~ (B, N_locs, Slot_dim)
            # attention equation [softmax(Q K^T) V]
            dots = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale  # dots ~ (B, N_slots, N_locs)
            attn = dots.softmax(dim=1) + self.epsilon  # enforcing competition between slots
            attn = attn / attn.sum(dim=-1, keepdim=True)  # attn ~ (B, N_slots, N_locs)
            self.attention_masks = attn
            updates = torch.einsum('b i d , b d j -> b i j', attn, v)  # updates ~ (B, N_slots, slot_dim)
            # further refinement
            slots = self.gru(
                updates.reshape(-1, self.dim_slots),
                slots_prev.reshape(-1, self.dim_slots)
            )
            slots = slots.reshape(B, -1, self.dim_slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def get_attention_masks(self):
        """
        Fetching last computer attention masks

        Returns:
        --------
        attention_masks: torch Tensor
            attention masks highligtinh the importance of each location to each slot
            Shape is (B, N_slots, N_locs)
        """
        B, N_slots, N_locs = self.attention_masks.shape
        masks = self.attention_masks
        return masks


class MetaAttention(nn.Module):
    """
    MetaClass for (Multi-Head) Key-Value Attention Mechanisms

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    out_dim: int/None
        Dimensionality of the output embeddings. If not given, it is set to 'emb_dim'
    """

    def __init__(self, emb_dim, num_heads=1, dropout=0., out_dim=None, **kwargs):
        """
        Initializer of the attention block
        """
        assert num_heads >= 1
        assert emb_dim % num_heads == 0, "Embedding dim. must be divisible by number of heads..."
        super().__init__()

        out_dim = out_dim if out_dim is not None else emb_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        # computing query, key, value for all embedding heads
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        # output projection
        self.out_projection = nn.Sequential(
                nn.Linear(emb_dim, out_dim, bias=False)
            )
        self.attention_masks = None
        return

    def forward(self, x):
        """ """
        raise NotImplementedError("Base-Class does not implement a 'forward' method...")

    def attention(self, query, key, value, dim_head):
        """
        Implementation of the standard normalized key-value attention equation
        """
        scale = dim_head ** -0.5  # 1/sqrt(d_k)
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale  # Q * K.T / sqrt(d_k)
        attention = dots.softmax(dim=-1)
        self.attention_masks = attention
        attention = self.drop(attention)
        vect = torch.einsum('b i d , b d j -> b i j', attention, value)  # Att * V
        return vect

    def get_attention_masks(self, reshape=None):
        """
        Fetching last computer attention masks
        """
        assert self.attention_masks is not None, "Attention masks have not yet been computed..."
        masks = self.attention_masks
        return masks


class MultiHeadSelfAttention(MetaAttention):
    """
    Vanilla Multi-Head dot-product attention mechanism.

    Args:
    -----
    emb_dim: integer
        Dimensionality of the token embeddings.
    num_heads: integer
        Number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact!
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    """

    def __init__(self, emb_dim, num_heads=8, dropout=0.):
        """
        Initializer of the attention block
        """
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        return

    def forward(self, x, **kwargs):
        """
        Forward pass through multi-head attention
        """
        batch_size, num_tokens, token_dim = x.size()
        dim_head = token_dim // self.num_heads

        # linear projections
        q, k, v = self.q(x), self.k(x), self.v(x)

        # split into heads and move to batch-size side:
        # (Batch, Token, Dims) --> (Batch, Heads, Token, HeadDim) --> (Batch* Heads, Token, HeadDim)
        q = q.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        q = q.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        k = k.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        k = k.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        v = v.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        v = v.reshape(batch_size * self.num_heads, num_tokens, dim_head)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=dim_head)
        # rearranging heads and recovering original shape
        vect = vect.reshape(batch_size, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        vect = vect.reshape(batch_size * num_tokens, self.num_heads * dim_head)

        # output projection
        y = self.out_projection(vect)
        y = y.reshape(batch_size, num_tokens, self.num_heads * dim_head)
        return y


class TransformerBlock(nn.Module):
    """
    Tranformer encoder block.
    This is used as predictor module in SAVi.

    Args:
    -----
    embed_dim: int
        Dimensionality of the input embeddings
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    pre_norm: bool
        If True, transformer computes the LayerNorm before attention and MLP.
        Otherwise, LayerNorm is used after the aforementaitoned layers
    """

    def __init__(self, embed_dim, num_heads, mlp_size, pre_norm=False):
        """
        Module initializer
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        assert num_heads >= 1

        # MHA
        self.attn = MultiHeadSelfAttention(
            emb_dim=embed_dim,
            num_heads=num_heads,
        )
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )
        # LayerNorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_model()
        return

    @torch.no_grad()
    def _init_model(self):
        """ Parameter initialization """
        init_xavier_(self)

    def forward(self, inputs):
        """
        Forward pass through transformer encoder block
        """
        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(inputs)
            x = self.attn(x)
            x = x + inputs

            y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = self.attn(inputs)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z


#
