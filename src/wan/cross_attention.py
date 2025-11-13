import torch.nn as nn
from src.wan.self_attention import WanSelfAttention
from src.attention import optimized_attention


class WanT2VCrossAttention(WanSelfAttention):
  def forward(self, x, context, transformer_options={}, **kwargs):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    # compute query, key, value
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    # compute attention
    x = optimized_attention(
      q, k, v, heads=self.num_heads, transformer_options=transformer_options
    )

    x = self.o(x)
    return x


class WanI2VCrossAttention(WanSelfAttention):
  def __init__(
    self,
    dim,
    num_heads,
    window_size=(-1, -1),
    qk_norm=True,
    eps=1e-6,
    operation_settings={},
  ):
    super().__init__(
      dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings
    )

    self.k_img = operation_settings.get("operations").Linear(
      dim,
      dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    self.v_img = operation_settings.get("operations").Linear(
      dim,
      dim,
      device=operation_settings.get("device"),
      dtype=operation_settings.get("dtype"),
    )
    # self.alpha = nn.Parameter(torch.zeros((1, )))
    self.norm_k_img = (
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

  def forward(self, x, context, context_img_len, transformer_options={}):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
    """
    context_img = context[:, :context_img_len]
    context = context[:, context_img_len:]

    # compute query, key, value
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)
    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = optimized_attention(
      q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options
    )
    # compute attention
    x = optimized_attention(
      q, k, v, heads=self.num_heads, transformer_options=transformer_options
    )

    # output
    x = x + img_x
    x = self.o(x)
    return x


WAN_CROSSATTENTION_CLASSES = {
  "t2v_cross_attn": WanT2VCrossAttention,
  "i2v_cross_attn": WanI2VCrossAttention,
}
