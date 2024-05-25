from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channle=None):
        super().__init__()
        if not mid_channle:
            mid_channle = out_channel
        self.double_conv = nn.Sequential(
            # 因为后面接BatchNorm，所以bias没用
            nn.Conv2d(
                in_channel,
                mid_channle,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channle),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channle,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 最大池化没有特征提取能力，丢失特征太多
        # 这里改成用3x3，步长为2的卷积进行下采样
        self.down_sample = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down_sample(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="biliner", align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
