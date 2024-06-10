from torch.utils import checkpoint
from unet_parts import DoubleConv, OutConv, UpSample, DownSample
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down_sample1 = DownSample(64)
        self.conv2 = DoubleConv(64, 128)
        self.down_sample2 = DownSample(128)
        self.conv3 = DoubleConv(128, 256)
        self.down_sample3 = DownSample(256)
        self.conv4 = DoubleConv(256, 512)
        self.down_sample4 = DownSample(512)
        self.conv5 = DoubleConv(512, 1024)
        self.up_sample1 = UpSample(1024)
        self.conv6 = DoubleConv(1024, 512)
        self.up_sample2 = UpSample(512)
        self.conv7 = DoubleConv(512, 256)
        self.up_sample3 = UpSample(256)
        self.conv8 = DoubleConv(256, 128)
        self.up_sample4 = UpSample(128)
        self.conv9 = DoubleConv(128, 64)
        self.out = OutConv(64, out_classes)

    def forward(self, x):
        run1 = self.conv1(x)
        run2 = self.conv2(self.down_sample1(run1))
        run3 = self.conv3(self.down_sample2(run2))
        run4 = self.conv4(self.down_sample3(run3))
        run5 = self.conv5(self.down_sample4(run4))
        run6 = self.conv6(self.up_sample1(run5))
        run7 = self.conv7(self.up_sample2(run6))
        run8 = self.conv8(self.up_sample3(run7))
        run9 = self.conv9(self.up_sample4(run8))
        final_run = self.out(run9)
        return final_run

    def use_checkpoint(self):
        """Save the memory, but increase the runnig time."""

        self.conv1 = checkpoint(self.conv1)
        self.down_sample1 = checkpoint(self.down_sample1)
        self.conv2 = checkpoint(self.conv2)
        self.down_sample2 = checkpoint(self.down_sample2)
        self.conv3 = checkpoint(self.conv3)
        self.down_sample3 = checkpoint(self.down_sample3)
        self.conv4 = checkpoint(self.conv4)
        self.down_sample4 = checkpoint(self.down_sample4)
        self.conv5 = checkpoint(self.conv5)
        self.up_sample1 = checkpoint(self.up_sample1)
        self.conv6 = checkpoint(self.conv6)
