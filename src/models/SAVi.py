"""
Implementation of the SAVi model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.logger import print_
from models import SlotAttention, SoftPositionEmbed, get_encoder, get_decoder,\
    TransformerBlock, get_initalizer
from models.model_utils import init_xavier_


class SAVi(nn.Module):
    """
    SAVi model as described in the paper:
        - Kipf, Thomas, et al. "Conditional object-centric learning from video." ICLR. 2022

    Args:
    -----
    resolution: list/tuple (int, int)
        spatial size of the input images
    num_slots: integer
        number of object slots to use. Corresponds to N-objects + background
    slot_dim: integer
        Dimensionality of the object slot embeddings
    num_iterations: integer
        number of recurrent iterations in Slot Attention for slot refinement.
    num_iterations_first: none/interger
        If specified, number of recurrent iterations for the first frame in the sequence. If not
        given, it is set to 'num_iterations'
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    encoder_type: string
        Name of the encoder to use
    num_channels: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer in the encoder
    decoder_resolution: list/tuple (int, int)
        spatial resolution of the decoder. If not the same as 'resolution', the
        decoder needs to use some padding/stride for upsampling
    initializer: string
        Type of intializer employed to initialize the slots at the first time step
    """

    def __init__(self, resolution, num_slots, slot_dim=64, num_iterations=3, in_channels=3,
                 kernel_size=5, encoder_type="ConvEncoder", num_channels=(32, 32, 32, 32),
                 downsample_encoder=False, downsample_decoder=False, upsample=4, decoder_resolution=(8, 8),
                 use_predictor=True, initializer="LearnedRandom", **kwargs):
        """ Model initializer """
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations_first = kwargs.get("num_iterations_first", num_iterations)
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.hidden_dims = num_channels
        self.decoder_resolution = decoder_resolution
        self.initializer_mode = initializer

        # building slot initializer
        print_("Initializer:")
        print_(f"  --> mode={initializer}")
        print_(f"  --> slot_dim={slot_dim}")
        print_(f"  --> num_slots={num_slots}")
        self.initializer = get_initalizer(
                mode=initializer,
                slot_dim=slot_dim,
                num_slots=num_slots
            )

        # Building convolutional encoder modules
        print_("Encoder:")
        print_(f"  --> Encoder_type={encoder_type}")
        print_(f"  --> Downsample_encoder={downsample_encoder}")
        print_(f"  --> in_channels={in_channels}")
        print_(f"  --> num_channels={num_channels}")
        print_(f"  --> kernel_size={kernel_size}")
        self.encoder = get_encoder(
                encoder_name=encoder_type,
                downsample_encoder=downsample_encoder,
                in_channels=in_channels,
                num_channels=num_channels,
                kernel_size=kernel_size
            )

        # postionwise encoder modules
        if encoder_type in ["ConvEncoder"]:
            self.out_features = self.hidden_dims[-1]
        else:
            self.out_features = self.encoder.out_features
        mlp_hidden = kwargs.get("mlp_hidden", 128)
        mlp_encoder_dim = kwargs.get("mlp_encoder_dim", self.out_features)
        self.encoder_pos_embedding = SoftPositionEmbed(
                hidden_size=self.out_features,
                resolution=resolution
            )
        self.encoder_mlp = nn.Sequential(
            nn.LayerNorm(self.out_features),
            nn.Linear(self.out_features, mlp_encoder_dim),
            nn.ReLU(),
            nn.Linear(mlp_encoder_dim, mlp_encoder_dim),
        )

        # recursive dynamics module
        self.predictor = TransformerBlock(
                embed_dim=slot_dim,
                num_heads=4,
                mlp_size=256
            ) if use_predictor else nn.Identity()

        # Building decoder modules
        print_("Decoder:")
        print_(f"  --> Resolution={resolution}")
        print_(f"  --> Num channelsl={num_channels}")
        print_(f"  --> Upsample={upsample}")
        print_(f"  --> Downsample_encoder={downsample_encoder}")
        print_(f"  --> Downsample_decoder={downsample_decoder}")
        print_(f"  --> Decoder_resolution={decoder_resolution}")
        self.decoder_pos_embedding = SoftPositionEmbed(
                hidden_size=slot_dim,
                resolution=self.decoder_resolution
            )
        num_channels = kwargs.get("num_channels_decoder", num_channels[:-1])
        self.decoder = get_decoder(
                decoder_name="ConvDecoder",
                in_channels=slot_dim,
                hidden_dims=num_channels,
                kernel_size=kernel_size,
                decoder_resolution=decoder_resolution,
                upsample=upsample if downsample_decoder else None
            )

        # slot attention
        self.slot_attention = SlotAttention(
            dim_feats=mlp_encoder_dim,
            dim_slots=slot_dim,
            num_slots=self.num_slots,
            num_iters_first=self.num_iterations_first,
            num_iters=self.num_iterations,
            mlp_hidden=mlp_hidden
        )

        self._init_model()
        return

    @torch.no_grad()
    def _init_model(self):
        """
        Initalization of the model parameters
        Adapted from:
            https://github.com/addtt/object-centric-library/blob/main/models/slot_attention/trainer.py
        """
        init_xavier_(self)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_ih)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_hh)
        torch.nn.init.orthogonal_(self.slot_attention.gru.weight_hh)
        if hasattr(self.slot_attention, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.slot_attention.dim_slots))
            torch.nn.init.uniform_(self.slot_attention.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slot_attention.slots_sigma, -limit, limit)
        return

    def forward(self, input, num_imgs=10, **kwargs):
        """
        Forward pass through the model

        Args:
        -----
        input: torch Tensor
            Images to process with SAVi. Shape is (B, NumImgs, C, H, W)
        num_imgs: int
            Number of images to recursively encode into object slots.

        Returns:
        --------
        slot_history: torch Tensor
            Object slots encoded at every time step. Shape is (B, num_imgs, num_slots, slot_dim)
        recons_history: torch Tensor
            Rendered video frames by decoding and combining the slots. Shape is (B, num_imgs, C, H, W)
        ind_recons_history: torch Tensor
            Rendered objects by decoding slots. Shape is (B, num_imgs, num_slots, C, H, W)
        masks_history: torch Tensor
            Rendered object masks by decoding slots. Shape is (B, num_imgs, num_slots, 1, H, W)
        """
        slot_history = []
        reconstruction_history = []
        individual_recons_history = []
        masks_history = []

        # initializing slots by randomly sampling them or encoding some representations (e.g. BBox)
        predicted_slots = self.initializer(batch_size=input.shape[0], **kwargs)

        # recursively mapping video frames into object slots
        for t in range(num_imgs):
            imgs = input[:, t]
            img_feats = self.encode(imgs)
            slots = self.apply_attention(img_feats, predicted_slots=predicted_slots, step=t)
            recon_combined, (recons, masks) = self.decode(slots)
            predicted_slots = self.predictor(slots)
            slot_history.append(slots)
            reconstruction_history.append(recon_combined)
            individual_recons_history.append(recons)
            masks_history.append(masks)

        slot_history = torch.stack(slot_history, dim=1)
        recons_history = torch.stack(reconstruction_history, dim=1)
        ind_recons_history = torch.stack(individual_recons_history, dim=1)
        masks_history = torch.stack(masks_history, dim=1)
        return slot_history, recons_history, ind_recons_history, masks_history

    def encode(self, input):
        """
        Encoding an image into image features
        """
        B, C, H, W = input.shape

        # encoding input frame and adding positional encodding
        x = self.encoder(input)  # x ~ (B,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos_embedding(x)  # x ~ (B,H,W,C)

        # further encodding with 1x1 Conv (implemented as shared MLP)
        x = torch.flatten(x, 1, 2)
        x = self.encoder_mlp(x)  # x ~ (B, N, Dim)
        return x

    def apply_attention(self, x, predicted_slots=None, step=0):
        """
        Applying slot attention on image features
        """
        slots = self.slot_attention(x, slots=predicted_slots, step=step)  # slots ~ (B, N_slots, Slot_dim)
        return slots

    def decode(self, slots):
        """
        Decoding slots into objects and masks
        """
        B, N_S, S_DIM = slots.shape

        # adding broadcasing for the dissentangled decoder
        slots = slots.reshape((-1, 1, 1, S_DIM))
        slots = slots.repeat(
                (1, self.decoder_resolution[0], self.decoder_resolution[1], 1)
            )  # slots ~ (B*N_slots, H, W, Slot_dim)

        # adding positional embeddings to reshaped features
        slots = self.decoder_pos_embedding(slots)  # slots ~ (B*N_slots, H, W, Slot_dim)
        slots = slots.permute(0, 3, 1, 2)

        y = self.decoder(slots)  # slots ~ (B*N_slots, Slot_dim, H, W)

        # recons and masks have shapes [B, N_S, C, H, W] & [B, N_S, 1, H, W] respectively
        y_reshaped = y.reshape(B, -1, self.in_channels + 1, y.shape[2], y.shape[3])
        recons, masks = y_reshaped.split([self.in_channels, 1], dim=2)

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, (recons, masks)


#
