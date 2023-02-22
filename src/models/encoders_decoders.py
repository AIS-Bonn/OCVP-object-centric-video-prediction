"""
Implementation of decoder modules
"""

import torch.nn as nn
import torch.nn.functional as F

from models.model_blocks import ConvBlock

ENCODERS = ["ConvEncoder"]
DECODERS = ["ConvDecoder", "UpsampleDecoder"]


def get_encoder(encoder_name, downsample_encoder, in_channels, num_channels, kernel_size, **kwargs):
    """
    Instanciating an encoder given the model name and parameters
    """
    if encoder_name not in ENCODERS:
        raise ValueError(f"Unknwon encoder_name {encoder_name}. Use one of {ENCODERS}")

    if(encoder_name == "ConvEncoder"):
        encoder_class = DownsamplingConvEncoder if downsample_encoder else SimpleConvEncoder
        encoder = encoder_class(
                in_channels=in_channels,
                hidden_dims=num_channels,
                kernel_size=kernel_size
            )
    else:
        raise NotImplementedError(f"Unknown encoder {encoder_name}...")

    return encoder


def get_decoder(decoder_name, **kwargs):
    """
    Instanciating a decoder given the model name and parameters
    """
    if decoder_name not in DECODERS:
        raise ValueError(f"Unknwon decoder_name {decoder_name}. Use one of {DECODERS}")

    if(decoder_name == "ConvDecoder"):
        decoder = Decoder(**kwargs)
    elif(decoder_name == "UpsampleDecoder"):
        decoder = UpsampleDecoder(**kwargs)
    else:
        raise NotImplementedError(f"Unknown decoder {decoder_name}...")

    return decoder


class SimpleConvEncoder(nn.Module):
    """
    Simple fully convolutional encoder

    Args:
    -----
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    """

    def __init__(self, in_channels=3, hidden_dims=(64, 64, 64, 64), kernel_size=5, **kwargs):
        """
        Module initializer
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.max_pool = kwargs.get("max_pool", None)

        self.encoder = self._build_encoder()
        return

    def _build_encoder(self):
        """
        Creating convolutional encoder given dimensionality parameters
        """
        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                    batch_norm=self.batch_norm,
                    max_pool=self.max_pool
                )
            modules.append(block)
            in_channels = h_dim
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, x):
        """ Forward pass """
        y = self.encoder(x)
        return y


class DownsamplingConvEncoder(nn.Module):
    """
    Convolutional encoder that dowsnamples images by factor of 4

    Args:
    -----
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    """

    DOWNSAMPLE = [0, 1, 2]

    def __init__(self, in_channels=3, hidden_dims=(64, 64, 64, 64), kernel_size=5, **kwargs):
        """
        Module initializer
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.max_pool = kwargs.get("max_pool", None)

        self.encoder = self._build_encoder()
        return

    def _build_encoder(self):
        """
        Creating convolutional encoder given dimensionality parameters
        """
        modules = []
        in_channels = self.in_channels
        for i, h_dim in enumerate(self.hidden_dims):
            # stride = 2 if i == mid or i == mid+1 else self.stride
            stride = 2 if i in DownsamplingConvEncoder.DOWNSAMPLE else self.stride
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=self.kernel_size // 2,
                    batch_norm=self.batch_norm,
                    max_pool=self.max_pool
                )
            modules.append(block)
            in_channels = h_dim
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, x):
        """
        Forward pass
        """
        y = self.encoder(x)
        return y


class Decoder(nn.Module):
    """
    Simple fully convolutional decoder

    Args:
    -----
    in_channels: int
        Number of input channels to the decoder
    hidden_dims: list
        List with the hidden dimensions in the decoder. Final value is the number of output channels
    kernel_size: int
        Kernel size for the convolutional layers
    upsample: int or None
        If not None, feature maps are upsampled by this amount after every hidden convolutional layer
    """

    def __init__(self, in_channels, hidden_dims, kernel_size=5, upsample=None, out_channels=4, **kwargs):
        """
        Module initializer
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[0]
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.upsample = upsample
        self.out_channels = out_channels

        self.decoder = self._build_decoder()
        return

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.hidden_dims[i]
            modules.append(block)
            if self.upsample is not None and i > 0:
                modules.append(Upsample(scale_factor=self.upsample))
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_channels,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, x):
        y = self.decoder(x)
        return y


class UpsampleDecoder(nn.Module):
    """
    Simple fully convolutional decoder that upsamples by 2 after every convolution

    Args:
    -----
    in_channels: int
        Number of input channels to the decoder
    hidden_dims: list
        List with the hidden dimensions in the decoder. Final value is the number of output channels
    kernel_size: int
        Kernel size for the convolutional layers
    """

    def __init__(self, in_channels, hidden_dims, kernel_size=5, out_channels=4, **kwargs):
        """
        Module initializer
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[0]
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.out_channels = out_channels

        self.decoder = self._build_decoder()
        return

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.hidden_dims[i]
            modules.append(block)
            modules.append(Upsample(scale_factor=2))
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_channels,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, x):
        y = self.decoder(x)
        return y


class Upsample(nn.Module):
    """
    Overriding the upsample class to avoid an error of nn.Upsample with large tensors
    """

    def __init__(self, scale_factor):
        """
        Module initializer
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Forward pass
        """
        y = F.interpolate(x.contiguous(), scale_factor=self.scale_factor, mode='nearest')
        return y

    def __repr__(self):
        """ """
        str = f"Upsample(scale_factor={self.scale_factor})"
        return str


#
