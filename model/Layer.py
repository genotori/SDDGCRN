import math

import torch
import torch.nn.functional as F
from torch import nn


class nconv(nn.Module):
    def __init__(self, beta=0.2):
        super(nconv, self).__init__()
        self.theta = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=True)
        self.weight = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False)  # ?
        self.beta = beta

    def forward(self, x, adj):
        nodes_num = adj.shape[-1]
        supports = torch.stack([torch.eye(nodes_num).to(x.device), adj])
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        res = F.relu(self.theta(x_g))
        H = torch.cat((x.unsqueeze(1), res), dim=1)
        res = self.weight(H)
        res = res.squeeze()
        return res