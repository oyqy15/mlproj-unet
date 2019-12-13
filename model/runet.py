import torch
import torch.nn as nn

from .resnet import resnet34
from .vanila import DoubleConv

class Rup(nn.Module):
    def __init__(self, c_in, c_out):
        super(Rup, self).__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 4, 2, 2)
        self.conv = DoubleConv(c_in // 4, c_out)
    def forward(self, x):
        x = self.up(x) # [B, C, H, W] -> [B, C/4, 2H, 2W]
        x = self.conv(x) # [B, C/4, 2H, 2W] -> [B, c_out, 2H, 2W]
        return x

class Res2Unet(nn.Module):
    def __init__(self, c_in, c_out):
        super(Res2Unet, self).__init__()
        self.res = resnet34(True)
        self.up1 = Rup(512, 512)
        self.up2 = Rup(256 + 512, 256)
        self.up3 = Rup(128 + 256, 128)
        self.up4 = Rup(64 + 128, 64)
        self.up5 = Rup(128, 64) # [B, 64, 320, 640]
        self.out = nn.Conv2d(64, c_out, 1)

    def forward(self, x):
        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x = self.res.relu(x)
        x_ = self.res.maxpool(x)

        x1 = self.res.layer1(x_)
        x2 = self.res.layer2(x1)
        x3 = self.res.layer3(x2)
        x4 = self.res.layer4(x3)

        u1 = self.up1(x4)
        u2 = self.up2(torch.cat([u1, x3], 1))
        u3 = self.up3(torch.cat([u2, x2], 1))
        u4 = self.up4(torch.cat([u3, x1], 1))
        u5 = self.up5(torch.cat([u4, x], 1))

        u = self.out(u5)
        return torch.sigmoid(u)