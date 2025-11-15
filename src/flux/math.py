import torch
from torch import Tensor
from einops import rearrange
from src.model_management import is_device_mps


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
  assert dim % 2 == 0
  if is_device_mps(pos.device):
    device = torch.device("cpu")

  scale = torch.linspace(
    0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device
  )
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
  out = torch.stack(
    [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
  )
  out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
  return out.to(dtype=torch.float32, device=pos.device)
