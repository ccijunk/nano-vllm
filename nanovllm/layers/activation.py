import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    # dynamic=True: token dim varies per continuous-batching step (see layernorm.py)
    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
