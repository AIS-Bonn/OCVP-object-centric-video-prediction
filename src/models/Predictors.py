"""
Implementation of predictor modules and wrapper functionalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.logger import print_

from models.model_blocks import PositionalEncoding


class PredictorWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies any predictor module on a sequence of data

    Args:
    -----
    exp_params: dict
        Dictionary containing the experiment parameters
    predictor: nn.Module
        Instanciated predictor module to wrap.
    """

    def __init__(self, exp_params, predictor):
        """
        Module initializer
        """
        super().__init__()
        self.exp_params = exp_params
        self.predictor = predictor

        # prediction parameters
        self.num_context = exp_params["training_prediction"]["num_context"]
        self.num_preds = exp_params["training_prediction"]["num_preds"]
        self.teacher_force = exp_params["training_prediction"]["teacher_force"]
        self.skip_first_slot = exp_params["training_prediction"]["skip_first_slot"]
        self.video_length = exp_params["training_prediction"]["sample_length"]

        # predictor model parameters
        # For transformers, we have a moving window over the input tokens ('input buffer size')
        self.predictor_name = exp_params["model"]["predictor"]["predictor_name"]
        if "Transformer" in self.predictor_name or "OCVP" in self.predictor_name:
            predictor_params = exp_params["model"]["predictor"][self.predictor_name]
            self.input_buffer_size = predictor_params.get("input_buffer_size", None)
            if self.input_buffer_size is None:
                print_(f"  --> {self.predictor_name} buffer size is 'None'. Setting it as {self.num_context}")
                self.input_buffer_size = self.num_context
            if self.input_buffer_size < self.num_context:
                print_(f"  --> {self.predictor_name}'s {self.input_buffer_size = } is too small.")
                print_(f"  --> Using {self.num_context} instead...")
            else:
                print_(f"  --> Using buffer size {self.input_buffer_size}...")
        return

    def forward(self, slot_history, condition=None):
        """
        Iterating over a sequence of slots, predicting the subsequent slots

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)
        condition: torch Tensor
            One condition for each frame on basis of which the predictor should make its prediction (optional).
            Shape is (B, num_frames, cond_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        self._is_teacher_force()
        if self.predictor_name == "LSTM":
            pred_slots = self.forward_lstm(slot_history)
        elif "Transformer" in self.predictor_name or "OCVP" in self.predictor_name:
            if "Cond" in self.predictor_name:
                pred_slots = self.forward_cond_transformer(slot_history, condition)
            else:
                pred_slots = self.forward_transformer(slot_history)
        else:
            raise ValueError(f"Unknown {self.predictor_name = }...")
        return pred_slots

    def forward_lstm(self, slot_history):
        """
        Forward pass through an LSTM-based predictor module

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        B, L, num_slots, slot_dim = slot_history.shape
        first_slot_idx = 1 if self.skip_first_slot else 0
        pred_slots = []

        # reshaping slots: (B, L, num_slots, slot_dim) --> (B * num_slots, L, slot_dim)
        slot_history_lstm_input = slot_history.permute(0, 2, 1, 3)
        slot_history_lstm_input = slot_history_lstm_input.reshape(B * num_slots, L, slot_dim)

        # using seed images to initialize the RNN predictor
        self.predictor.init_hidden(b_size=B * num_slots, device=slot_history.device)
        for t in range(first_slot_idx, self.num_context - 1):
            _ = self.predictor(slot_history_lstm_input[:, t])

        # Autoregressively predicting the future slots
        next_input = slot_history_lstm_input[:, self.num_context - 1]
        for t in range(self.num_preds):
            cur_preds = self.predictor(next_input)
            next_input = slot_history_lstm_input[:, self.num_context+t] if self.teacher_force else cur_preds
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B * num_slots, num_preds, slot_dim)

        # reshaping back to (B, num_preds, num_slots, slot_dim)
        pred_slots = pred_slots.reshape(B, num_slots, self.num_preds, slot_dim).permute(0, 2, 1, 3)
        return pred_slots

    def forward_transformer(self, slot_history):
        """
        Forward pass through any Transformer-based predictor module

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        first_slot_idx = 1 if self.skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:self.num_context].clone()  # inial token buffer

        pred_slots = []
        for t in range(self.num_preds):
            cur_preds = self.predictor(predictor_input)[:, -1]  # get predicted slots from step
            next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots

    def forward_cond_transformer(self, slot_history, condition):
        """
        Forward pass through any conditional Transformer-based predictor module

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)
        condition: torch Tensor
            One condition for each frame on basis of which the predictor should make its prediction.
            Shape is (B, num_frames, cond_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        first_slot_idx = 1 if self.skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:self.num_context].clone()  # initial token buffer

        pred_slots = []
        for t in range(self.num_preds):
            cur_preds = self.predictor(predictor_input, condition[:, self.num_context-1+t].clone())[:, -1]  # get predicted slots from step
            next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots

    def _is_teacher_force(self):
        """
        Updating the teacher force value, depending on the training stage
            - In eval-mode, then teacher-forcing is always false
            - In train-mode, then teacher-forcing depends on the predictor parameters
        """
        if self.predictor.train is False:
            self.teacher_force = False
        else:
            self.teacher_force = self.exp_params["training_prediction"]["teacher_force"]
        return

    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs


class LSTMPredictor(nn.Module):
    """
    LSTM for predicting the (n+1)th object slot given a sequence of n slots

    Args:
    -----
    slot_dim: integer
        dimensionality of the slots
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        determine number of lstm cells
    residual: bool
        If True, a residual connection bridges across the predictor module
    """

    def __init__(self, slot_dim=64, hidden_dim=64, num_layers=2, residual=True):
        """
        Module initializer
        """
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual

        self.lstm = nn.ModuleList([])
        for n in range(num_layers):
            dim = slot_dim if n == 0 else hidden_dim
            self.lstm.append(nn.LSTMCell(input_size=dim, hidden_size=hidden_dim))

        self.init_hidden()
        return

    def forward(self, x):
        """
        Forward pass through LSTM predictor model

        Args:
        -----
        x: torch Tensor
            Current sequence element fed to the RNN. Shape is (B, Dim)

        Returns:
        --------
        output: torch Tensor
            Predicted next element in the sequence. Shape is (B, Dim)
        """
        input = x
        for i in range(self.num_layers):
            h, c = self.hidden_state[i]
            next_h, next_c = self.lstm[i](input, (h, c))
            self.hidden_state[i] = (next_h, next_c)
            input = self.hidden_state[i][0]

        output = input + x if self.residual else input
        return output

    def init_hidden(self, b_size=1, device=None):
        """
        Initializing hidden and cell states
        """
        hidden_state = []
        for _ in range(self.num_layers):
            cur_state = (torch.zeros(b_size, self.hidden_dim), torch.zeros(b_size, self.hidden_dim))
            hidden_state.append(cur_state)
        if device is not None:
            hidden_state = [(h[0].to(device), h[1].to(device)) for h in hidden_state]
        self.hidden_state = hidden_state
        return


class VanillaTransformerPredictor(nn.Module):
    """
    Vanilla Transformer Predictor module.
    It performs self-attention over all slots in the input buffer, jointly modelling
    the relational and temporal dimensions.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256,
                 num_layers=2, n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size
        print_("Instanciating Vanilla Transformer Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # MLPs to map slot-dim into token-dim and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim is split across num_heads, i.e., each head will have dimension embed_dim // num_heads)
        self.transformer_encoders = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        # Custom temrpoal encoding. All slots from the same time step share the encoding
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        return

    def forward(self, inputs):
        """
        Foward pass through the transformer predictor module to predic the subsequent object slots

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # mapping slots to tokens, and applying temporal positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through transformer encoder blocks
        token_output = time_encoded_input.reshape(B, num_imgs * num_slots, self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)
        token_output = token_output.reshape(B, num_imgs, num_slots, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output


class CondTransformerPredictor(nn.Module):
    """
    Conditonal Transformer Predictor module.
    In addition, this one gets a condition, e.g., action performed by an agent, for its prediction.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    cond_dim: int
        Dimensionality of condition input.
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs, cond_dim, token_dim=128, hidden_dim=256,
                 num_layers=2, n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.cond_dim = cond_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size
        print_("Instantiating Conditional Transformer Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> cond_dim: {self.cond_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # MLPs to map slot-dim into token-dim and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim is split across num_heads, i.e., each head will have dimension embed_dim // num_heads)
        self.transformer_encoders = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        # Custom temporal encoding. All slots from the same time step share the encoding
        # One is added to the input_buffer_size to account for the positional encoding of the input condition
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)  # +1)
        # Batch normalization for action
        # self.condition_norm = nn.BatchNorm1d(self.cond_dim)
        # Token embedding for action
        self.condition_embedding = nn.Linear(self.cond_dim, token_dim)

        return

    def forward(self, inputs, condition):
        """
        Foward pass through the transformer predictor module to predict the subsequent object slots and reward

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)
        condition: torch Tensor
            Condition the transformer output should be conditioned on, e.g., action performed by an agent. Shape is (B, cond_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """

        B, num_imgs, num_slots, slot_dim = inputs.shape

        # mapping slots to tokens, and applying temporal positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # embed condition
        # normalized_condition = self.condition_norm(condition)
        condition_token = self.condition_embedding(condition)
        # time_encoded_condition = self.pe.forward_cond(
        #        x=condition_token,
        #        batch_size=B
        #    )

        # feeding through transformer encoder blocks
        token_output = torch.cat((time_encoded_input.reshape(B, num_imgs * num_slots, self.token_dim), condition_token.unsqueeze(1)), dim=1)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)
        # throw away transformer output for condition token
        token_output = token_output[:, :-1, :].reshape(B, num_imgs, num_slots, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output


class OCVPSeq(nn.Module):
    """
    Sequential Object-Centric Video Prediction Transformer Module (OCVP-Seq).
    This module models the temporal dynamics and object interactions in a decoupled manner by
    sequentially applying object- and time-attention, i.e. [time, obj, time, ...]

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5):
        """
        Module Initialzer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating OCVP-Seq Predictor Module:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPSeqLayer(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads
                ) for _ in range(num_layers)]
            )

        # custom temporal encoding. All slots from the same time step share the same encoding
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        return

    def forward(self, inputs):
        """
        Forward pass through OCVP-Seq

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through OCVP-Seq transformer blocks
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output


class OCVPSeqLayer(nn.Module):
    """
    Sequential Object-Centric Video Prediction (OCVP-Seq) Transformer Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.nhead = n_heads

        self.object_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        return

    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out


class OCVPPar(nn.Module):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Predictor Module.
    This module models the temporal dynamics and object interactions in a dissentangled manner by
    applying relational- and temporal-attention in parallel.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating OCVP-Par Predictor Module:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim will be split across num_heads, i.e. each head will have dim embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPParLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        return

    def forward(self, inputs):
        """
        Forward pass through Object-Centric Transformer v1

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding tokens through transformer la<ers
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # projecting back to slot-dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output


class OCVPParLayer(nn.TransformerEncoderLayer):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Module.
    This module models the temporal dynamics and object interactions in a dissentangled manner by
    applying object- and time-attention in parallel.

    Args:
    -----
    d_model: int
        Dimensionality of the input tokens
    nhead: int
        Number of heads in multi-head attention
    dim_feedforward: int
        Hidden dimension in the MLP
    dropout: float
        Amount of dropout to apply. Default is 0.1
    activation: int
        Nonlinear activation in the MLP. Default is ReLU
    layer_norm_eps: int
        Epsilon value in the layer normalization components
    batch_first: int
        If True, shape is (B, num_tokens, token_dim); otherwise, it is (num_tokens, B, token_dim)
    norm_first: int
        If True, transformer is in mode pre-norm: otherwise, it is post-norm
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=True, device=None, dtype=None):
        """
        Module initializer
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=dtype
            )

        self.self_attn_obj = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        self.self_attn_time = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        return

    def forward(self, src, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-v2.
        Overloads PyTorch's transformer forward pass.

        Args:
        -----
        src: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), time_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, time_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, time_mask):
        """
        Forward pass through the parallel attention branches
        """
        B, num_imgs, num_slots, dim = x.shape

        # object-attention
        x_aux = x.clone().view(B * num_imgs, num_slots, dim)
        x_obj = self.self_attn_obj(
                query=x_aux,
                key=x_aux,
                value=x_aux,
                need_weights=False
            )[0]
        x_obj = x_obj.view(B, num_imgs, num_slots, dim)

        # time-attention
        x = x.transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
        x_time = self.self_attn_time(
                query=x,
                key=x,
                value=x,
                attn_mask=time_mask,
                need_weights=False
            )[0]
        x_time = x_time.view(B, num_slots, num_imgs, dim).transpose(1, 2)

        y = self.dropout1(x_obj + x_time)
        return y


#
