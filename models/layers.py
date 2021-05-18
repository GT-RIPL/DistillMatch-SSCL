import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class CosineScaling(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Add BN and linear layers
        Note: just for MLP at the moment. need bn2d compatible for cnn
        """
        super(CosineScaling, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.end_linear = nn.Linear(in_features, out_features)
        self.scale_linear = nn.Linear(in_features, 1, bias=False)
        self.fc_w = nn.Parameter(self.end_linear.weight)
        self.cossin_layer = CenterCosineSimilarity
        self.bn_scale = nn.BatchNorm1d(1) 

    def forward(self, x):

        scale = self.scale_linear(x)
        scale = self.bn_scale(scale)
        scale = torch.exp(scale)
        torch.transpose(scale,0,1)
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.fc_w)
        w_norm_transposed = torch.transpose(w_norm, 0, 1)
        x_cos = torch.mm(x_norm, w_norm_transposed)
        x_scaled = scale * x_cos
        return x_scaled


class CenterCosineSimilarity(nn.Module):
    def __init__(self, feat_dim, num_centers, eps=1e-8):
        super(CenterCosineSimilarity, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        self.eps = eps

    def forward(self, feat):
        norm_f = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_normalized = torch.div(feat, norm_f)
        norm_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        center_normalized = torch.div(self.centers, norm_c)
        return torch.mm(feat_normalized, center_normalized.t())