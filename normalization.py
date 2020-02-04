import torch
import torch.nn as nn
import torch.nn.functional as F

class PN(nn.Module):
    def __init__(self, norm_nc, label_nc, upsampling):
        super().__init__()

        self.upsampling = upsampling
        self.batch_norm = nn.BatchNorm2d(norm_nc, affine=False)

        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_scale = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_bias = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, pre_feat):

        normalized = self.batch_norm(x)

        pre_feat = F.interpolate(pre_feat, size=x.size()[2:], mode=self.upsampling)
        actv = self.mlp_shared(pre_feat)
        scale = self.mlp_scale(actv)
        bias = self.mlp_bias(actv)

        out = normalized * (1 + scale) + bias

        return out
