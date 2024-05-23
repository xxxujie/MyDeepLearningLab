from torchvision import nn


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, 3, 1, 1, padding_mode="reflect", bias=False
            ),
        )
