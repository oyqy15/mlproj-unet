import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c_in, c_out)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, c_in, c_out, upsample=False):
        super(Up, self).__init__()
        if upsample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(c_in, c_in // 2, 1),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, 2)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x1, x2):
        # x1: lower
        # x2: upper
        x1 = self.up(x1)
        diffH = x2.shape[2] - x1.shape[2]
        diffW = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1,
                   [diffW // 2, diffW - diffW // 2,
                    diffH // 2, diffH - diffH // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, c_in, c_out):
        super(Unet, self).__init__()
        self.c1 = DoubleConv(c_in, 64)
        self.c2 = Down(64, 128)
        self.c3 = Down(128, 256)
        self.c4 = Down(256, 512)
        self.c5 = Down(512, 1024)
        self.u1 = Up(1024, 512)
        self.u2 = Up(512, 256)
        self.u3 = Up(256, 128)
        self.u4 = Up(128, 64)
        self.out = nn.Conv2d(64, c_out, 1)

    def forward(self, x):
        # x: [bs, c_in, h, w]
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        u = self.u1(x5, x4)
        u = self.u2(u, x3)
        u = self.u3(u, x2)
        u = self.u4(u, x1)
        u = self.out(u)
        # u: [bs, c_out, h, w]
        return torch.sigmoid(u)
