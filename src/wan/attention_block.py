import torch
import torch.nn as nn
from src.wan.self_attention import WanSelfAttention
from src.model_management import cast_to
from src.wan.cross_attention import WAN_CROSSATTENTION_CLASSES


def repeat_e(e, x):
  repeats = 1
  if e.size(1) > 1:
    repeats = x.size(1) // e.size(1)
  if repeats == 1:
    return e
  if repeats * e.size(1) == x.size(1):
    return torch.repeat_interleave(e, repeats, dim=1)
  else:
    return torch.repeat_interleave(e, repeats + 1, dim=1)[:, : x.size(1)]


class WanAttentionBlock(nn.Module):
  def __init__(
    self,
    cross_attn_type,
    dim,
    ffn_dim,
    num_heads,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=False,
    eps=1e-6,
    operation_settings={},
  ):
    super().__init__()
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.num_heads = num_heads
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.cross_attn_norm = cross_attn_norm
    self.eps = eps

    # layers
    self.norm1 = operation_settings.get("operations").LayerNorm(
      dim,
      eps,
      elementwise_affine=False,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.self_attn = WanSelfAttention(
      dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings
    )
    self.norm3 = (
      operation_settings.get("operations").LayerNorm(
        dim,
        eps,
        elementwise_affine=True,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      )
      if cross_attn_norm
      else nn.Identity()
    )
    self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
      dim, num_heads, (-1, -1), qk_norm, eps, operation_settings=operation_settings
    )
    self.norm2 = operation_settings.get("operations").LayerNorm(
      dim,
      eps,
      elementwise_affine=False,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.ffn = nn.Sequential(
      operation_settings.get("operations").Linear(
        dim,
        ffn_dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
      nn.GELU(approximate="tanh"),
      operation_settings.get("operations").Linear(
        ffn_dim,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
    )

    # modulation
    self.modulation = nn.Parameter(
      torch.empty(
        1,
        6,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      )
    )

  def forward(
    self,
    x,
    e,
    freqs,
    context,
    context_img_len=257,
    transformer_options={},
  ):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, 6, C]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    # assert e.dtype == torch.float32

    if e.ndim < 4:
      e = (cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
    else:
      e = (
        cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e
      ).unbind(2)
    # assert e[0].dtype == torch.float32

    # self-attention
    x = x.contiguous()  # otherwise implicit in LayerNorm
    y = self.self_attn(
      torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
      freqs,
      transformer_options=transformer_options,
    )

    x = torch.addcmul(x, y, repeat_e(e[2], x))
    del y

    # cross-attention & ffn
    x = x + self.cross_attn(
      self.norm3(x),
      context,
      context_img_len=context_img_len,
      transformer_options=transformer_options,
    )
    y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
    x = torch.addcmul(x, y, repeat_e(e[5], x))
    return x
