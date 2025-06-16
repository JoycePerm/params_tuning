import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    def __init__(self, h_dim, c_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(c_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.norm = nn.LayerNorm(h_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, c=None):
        gamma, beta = torch.chunk(self.mlp(c), 2, dim=-1)

        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        x = self.norm(x) * (1 + gamma) + beta

        return x
