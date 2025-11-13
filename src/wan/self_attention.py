import torch.nn as nn
from src.math import apply_rope1
from src.attention import optimized_attention


class WanSelfAttention(nn.Module):
  def __init__(
    self,
    dim,
    num_heads,
    window_size=(-1, -1),
    qk_norm=True,
    eps=1e-6,
    kv_dim=None,
    operation_settings={},
  ):
    assert dim % num_heads == 0
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.eps = eps
    if kv_dim is None:
      kv_dim = dim

    # layers
    self.q = operation_settings.get("operations").Linear(
      dim,
      dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.k = operation_settings.get("operations").Linear(
      kv_dim,
      dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.v = operation_settings.get("operations").Linear(
      kv_dim,
      dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.o = operation_settings.get("operations").Linear(
      dim,
      dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.norm_q = (
      operation_settings.get("operations").RMSNorm(
        dim,
        eps=eps,
        elementwise_affine=True,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      )
      if qk_norm
      else nn.Identity()
    )
    self.norm_k = (
      operation_settings.get("operations").RMSNorm(
        dim,
        eps=eps,
        elementwise_affine=True,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      )
      if qk_norm
      else nn.Identity()
    )

  def forward(self, x, freqs, transformer_options={}):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    def qkv_fn_q(x):
      q = self.norm_q(self.q(x)).view(b, s, n, d)
      return apply_rope1(q, freqs)

    def qkv_fn_k(x):
      k = self.norm_k(self.k(x)).view(b, s, n, d)
      return apply_rope1(k, freqs)

    # These two are VRAM hogs, so we want to do all of q computation and
    # have pytorch garbage collect the intermediates on the sub function
    # return before we touch k
    q = qkv_fn_q(x)
    k = qkv_fn_k(x)

    x = optimized_attention(
      q.view(b, s, n * d),
      k.view(b, s, n * d),
      self.v(x).view(b, s, n * d),
      heads=self.num_heads,
      transformer_options=transformer_options,
    )

    x = self.o(x)
    return x
