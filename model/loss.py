import torch
import torch.nn as nn


def dice_co(X, Y, eps=1e-7):
    # 2 |X and Y| / (|X| + |Y|)
    # X.shape (bs, c, h, w), float
    # Y.shape (bs, c, h, w), float
    tp = torch.sum(X * Y)
    return (2 * tp + eps) / (torch.sum(X) + torch.sum(Y) + eps)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7, threshold=None):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.threshold = threshold
    def forward(self, x, y):
        # x: from unet
        # y: from training data
        if not(self.threshold is None):
            x = (x > self.threshold).float()
        return 1 - dice_co(x, y, self.eps)

class GDiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(GDiceLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        # x: from unet
        # y: from training data
        # (bs, c, h, w)

        w = torch.sum(y, [0, 2, 3])
        w = 1 / (w * w  + self.eps)
        up = torch.sum(w * torch.sum(x * y, [0, 2, 3]))
        dn = torch.sum(x, [0, 2, 3]) + torch.sum(y, [0, 2, 3])
        dn = torch.sum(w * dn)
        return 1 - 2 * (up + self.eps) / (dn + self.eps) 

class BceDiceLoss(DiceLoss):
    def __init__(self, lambda_bce=1.0, lambda_dice=1.0, eps=1e-7, threshold=None):
        super().__init__(eps, threshold)
        self.bce = nn.BCELoss()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
    def forward(self, x, y):
        # x: from unet
        # y: from training data
        dice_loss = super().forward(x, y)
        bce_loss = self.bce(x, y)
        return self.lambda_bce * bce_loss + self.lambda_dice * dice_loss
     
