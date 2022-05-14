import torch
from torch import nn


class Conv(nn.Module):
    """A Unet Convolution module."""

    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input):
        """Convolution forward.

        Args:
            input (Tensor): with shape [Batch, InChannel, H, W]

        Returns:
            Tensor: with shape [Batch, OutChannel, H, W]
        """
        input = self.lrelu(self.conv1(input))
        input = self.lrelu(self.conv2(input))
        return input


class UpSample(nn.Module):
    """A Unet up sample module."""

    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            channel_in, channel_out, kernel_size=2, stride=2, bias=False
        )

    def forward(self, input, conv_result):
        """Up Sample module.

        Args:
            input (Tensor): with shape [Batch, InChannel, H, W]
            conv_result (Tensor): with shape [Batch, OutChannel, 2H, 2W]

        Returns:
            Tensor: with shape [Batch, 2 * OutChannel, 2H, 2W]
        """
        input = self.upsample(input)
        input = torch.cat((input, conv_result), dim=1)
        return input


class Unet(nn.Module):
    """Unet"""

    def __init__(self, channel_in: int, channel_out: int) -> None:
        super().__init__()
        self.conv_down1 = Conv(channel_in=channel_in, channel_out=32)
        self.conv_down2 = Conv(channel_in=32, channel_out=64)
        self.conv_down3 = Conv(channel_in=64, channel_out=128)
        self.conv_down4 = Conv(channel_in=128, channel_out=256)
        self.conv_down5 = Conv(channel_in=256, channel_out=512)
        self.down_sample = nn.MaxPool2d(kernel_size=2)

        self.sample_up1 = UpSample(channel_in=512, channel_out=256)
        self.conv_up1 = Conv(channel_in=512, channel_out=256)

        self.sample_up2 = UpSample(channel_in=256, channel_out=128)
        self.conv_up2 = Conv(channel_in=256, channel_out=128)

        self.sample_up3 = UpSample(channel_in=128, channel_out=64)
        self.conv_up3 = Conv(channel_in=128, channel_out=64)

        self.sample_up4 = UpSample(channel_in=64, channel_out=32)
        self.conv_up4 = Conv(channel_in=64, channel_out=32)

        self.conv_result = nn.Conv2d(32, channel_out, kernel_size=1)

    def forward(self, input):
        """Unet forward.

        Args:
            input (Tensor): with shape [Batch, InChannel, H, W]

        Returns:
            Tensor: with shape [Batch, OutChannel, H, W]
        """
        input = self.conv_down1(input)
        conv1 = input  # channel = 32
        input = self.down_sample(input)

        input = self.conv_down2(input)
        conv2 = input  # channel = 64
        input = self.down_sample(input)

        input = self.conv_down3(input)
        conv3 = input  # channel = 128
        input = self.down_sample(input)

        input = self.conv_down4(input)
        conv4 = input  # channel = 256
        input = self.down_sample(input)

        output = self.conv_down5(input)

        output = self.sample_up1(output, conv4)
        output = self.conv_up1(output)  # channel = 256

        output = self.sample_up2(output, conv3)
        output = self.conv_up2(output)  # channel = 128

        output = self.sample_up3(output, conv2)
        output = self.conv_up3(output)  # channel = 64

        output = self.sample_up4(output, conv1)
        output = self.conv_up4(output)  # channel = 32

        output = self.conv_result(output)
        return output


if __name__ == "__main__":
    # test
    input = torch.rand((2, 4, 512, 512))
    net = Unet(4, 10)
    output = net(input)
    print(output.shape)
    # [2, 10, 512, 512]
