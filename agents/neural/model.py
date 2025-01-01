import torch
from torch import nn
from torch.nn import functional as F

from agents.neural.utils import NetworkOutput, positional_encoding


def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 middle_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvRelu(in_channels, middle_channels)
        self.conv2 = ConvRelu(middle_channels, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = skip + x
        return x


class DecoderBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def mlp(in_channels, out_channels, middle_channels):
    return nn.Sequential(
        nn.Linear(in_channels, middle_channels),
        nn.ReLU(),
        nn.Linear(middle_channels, out_channels)
    )


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        is_deconv: bool = False,
    ):

        super().__init__()

        self.pe = torch.tensor(positional_encoding(24, 24)).float()  # (1, 1, 24, 24))

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 2*num_filters, 1),
            self.relu,
            nn.Conv2d(2*num_filters, num_filters, 1),
        )

        self.conv1 = nn.Sequential(
           EncoderBlock(num_filters, 2*num_filters, num_filters),
           EncoderBlock(num_filters, 2*num_filters, 2*num_filters)
        )

        self.conv2 = nn.Sequential(
           EncoderBlock(2*num_filters, 4*num_filters, 2*num_filters),
           EncoderBlock(2*num_filters, 4*num_filters, 4*num_filters)
        )

        self.conv3 = nn.Sequential(
           EncoderBlock(4*num_filters, 8*num_filters, 4*num_filters),
           EncoderBlock(4*num_filters, 8*num_filters, 8*num_filters)
        )

        self.center = DecoderBlockV2(
            8*num_filters, 16*num_filters, 8*num_filters, is_deconv
        )

        self.dec3 = DecoderBlockV2(
            num_filters * 8 * 2, num_filters * 4 * 2, num_filters * 4, is_deconv
        )
        self.dec2 = DecoderBlockV2(
            num_filters * 4 * 2, num_filters * 2 * 2, num_filters, is_deconv
        )
        self.dec1 = ConvRelu(3*num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, 5 + 2, kernel_size=1)

        self.value = mlp(8*num_filters, 1, num_filters)

    def forward(self, x: torch.Tensor) -> NetworkOutput:
        conv0 = self.conv0(x) + self.pe
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))

        center = self.center(self.pool(conv3))

        # global max pooling
        v = F.max_pool2d(center, kernel_size=center.size()[2:])[:, :, 0, 0]

        dec3 = self.dec3(torch.cat([center, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        logits = self.final(dec1)

        m, s = F.softmax(logits[:, :5], 1), F.softmax(logits[:, -2:], 1)
        v = F.tanh(self.value(v))

        return NetworkOutput(v, m, s)
