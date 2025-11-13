import torch
import torch.nn as nn
from torch import Tensor
from src.flux.math import rope


class EmbedND(nn.Module):
  def __init__(self, dim: int, theta: int, axes_dim: list):
    super().__init__()
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def forward(self, ids: Tensor) -> Tensor:
    n_axes = ids.shape[-1]
    emb = torch.cat(
      [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
      dim=-3,
    )

    return emb.unsqueeze(1)
