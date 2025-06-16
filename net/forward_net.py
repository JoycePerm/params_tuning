import torch
import torch.nn as nn
import math
from net.adaLN import AdaLayerNorm


class Embedder(nn.Module):
    def __init__(self, x_dim, c_dim, h_dim):
        super().__init__()
        self.x_dim = x_dim
        self.c = c_dim
        self.h_dim = h_dim

        self.x_emb = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.c_emb = nn.Sequential(
            nn.Linear(c_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            # nn.Linear(h_dim, h_dim),
        )
        self.ln = nn.LayerNorm(h_dim)
        # self.adaln = AdaLayerNorm(h_dim, c_dim)

    def forward(self, x, c):

        x_emb = self.x_emb(x)
        c_scale, c_shift = self.c_emb(c).chunk(2, dim=-1)
        # emb = x_emb * c_scale + c_shift
        emb = self.ln(x_emb) * c_scale + c_shift
        # emb = self.adaln(x_emb, c) * c_scale + c_shift

        return emb


class ContNet(nn.Module):
    def __init__(self, x_dim, c_dim, y_dim, h_dim):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.y_dim = y_dim
        self.h_dim = h_dim

        self.embedder = Embedder(x_dim, c_dim, h_dim)
        # self.act = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 2 * h_dim),
            nn.BatchNorm1d(2 * h_dim),
            nn.ReLU(),
            nn.Linear(2 * h_dim, 2 * h_dim),
            nn.BatchNorm1d(2 * h_dim),
            nn.ReLU(),
            nn.Linear(2 * h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim),
        )

    def forward(self, x, c):
        # c = c.to(torch.int32)
        # print(x, c)
        # emb = self.act(self.embedder(x, c))
        emb = self.embedder(x, c)
        y_pred = self.mlp(emb)

        return y_pred
