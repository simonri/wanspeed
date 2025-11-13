import torch
import torch.nn as nn
import math
from src.model_management import cast_to
from src.wan.attention_block import repeat_e


class Head(nn.Module):
  def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
    super().__init__()
    self.dim = dim
    self.out_dim = out_dim
    self.patch_size = patch_size
    self.eps = eps

    # layers
    out_dim = math.prod(patch_size) * out_dim
    self.norm = operation_settings.get("operations").LayerNorm(
      dim,
      eps,
      elementwise_affine=False,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.head = operation_settings.get("operations").Linear(
      dim,
      out_dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )

    # modulation
    self.modulation = nn.Parameter(
      torch.empty(
        1,
        2,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      )
    )

  def forward(self, x, e):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        e(Tensor): Shape [B, C]
    """
    # assert e.dtype == torch.float32
    if e.ndim < 3:
      e = (
        cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)
      ).chunk(2, dim=1)
    else:
      e = (
        cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0)
        + e.unsqueeze(2)
      ).unbind(2)

    x = self.head(torch.addcmul(repeat_e(e[0], x), self.norm(x), 1 + repeat_e(e[1], x)))
    return x
