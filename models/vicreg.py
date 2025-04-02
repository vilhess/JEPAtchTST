import torch 
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    def forward(self, x):
        x = x.flatten(start_dim=1)
        f = x.shape[1]
        x = x - x.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        loss1 = torch.mean(F.relu(1 - std_x))
        loss2 = self._off_diagonal(cov_x).pow_(2).sum().div(f)
        return loss1, loss2