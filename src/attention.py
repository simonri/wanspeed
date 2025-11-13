import functools
import torch
from einops import rearrange, repeat, einsum
import src.model_management as model_management
import src.ops as ops


def get_attn_precision(attn_precision, current_dtype):
  return attn_precision


def exists(val):
  return val is not None


def default(val, d):
  if exists(val):
    return val
  return d


def wrap_attn(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    remove_attn_wrapper_key = False
    try:
      if "_inside_attn_wrapper" not in kwargs:
        transformer_options = kwargs.get("transformer_options", None)
        remove_attn_wrapper_key = True
        kwargs["_inside_attn_wrapper"] = True
        if transformer_options is not None:
          if "optimized_attention_override" in transformer_options:
            return transformer_options["optimized_attention_override"](
              func, *args, **kwargs
            )
      return func(*args, **kwargs)
    finally:
      if remove_attn_wrapper_key:
        del kwargs["_inside_attn_wrapper"]

  return wrapper


@wrap_attn
def attention_basic(
  q,
  k,
  v,
  heads,
  mask=None,
  attn_precision=None,
  skip_reshape=False,
  skip_output_reshape=False,
  **kwargs,
):
  attn_precision = get_attn_precision(attn_precision, q.dtype)

  if skip_reshape:
    b, _, _, dim_head = q.shape
  else:
    b, _, dim_head = q.shape
    dim_head //= heads

  scale = dim_head**-0.5

  h = heads
  if skip_reshape:
    q, k, v = map(
      lambda t: t.reshape(b * heads, -1, dim_head),
      (q, k, v),
    )
  else:
    q, k, v = map(
      lambda t: t.unsqueeze(3)
      .reshape(b, -1, heads, dim_head)
      .permute(0, 2, 1, 3)
      .reshape(b * heads, -1, dim_head)
      .contiguous(),
      (q, k, v),
    )

  # force cast to fp32 to avoid overflowing
  if attn_precision == torch.float32:
    sim = einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
  else:
    sim = einsum("b i d, b j d -> b i j", q, k) * scale

  del q, k

  if exists(mask):
    if mask.dtype == torch.bool:
      mask = rearrange(
        mask, "b ... -> b (...)"
      )  # TODO: check if this bool part matches pytorch attention
      max_neg_value = -torch.finfo(sim.dtype).max
      mask = repeat(mask, "b j -> (b h) () j", h=h)
      sim.masked_fill_(~mask, max_neg_value)
    else:
      if len(mask.shape) == 2:
        bs = 1
      else:
        bs = mask.shape[0]
      mask = (
        mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1])
        .expand(b, heads, -1, -1)
        .reshape(-1, mask.shape[-2], mask.shape[-1])
      )
      sim.add_(mask)

  # attention, what we cannot get enough of
  sim = sim.softmax(dim=-1)

  out = einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)

  if skip_output_reshape:
    out = out.unsqueeze(0).reshape(b, heads, -1, dim_head)
  else:
    out = (
      out.unsqueeze(0)
      .reshape(b, heads, -1, dim_head)
      .permute(0, 2, 1, 3)
      .reshape(b, -1, heads * dim_head)
    )
  return out


SDP_BATCH_LIMIT = 2**15


@wrap_attn
def attention_pytorch(
  q,
  k,
  v,
  heads,
  mask=None,
  attn_precision=None,
  skip_reshape=False,
  skip_output_reshape=False,
  **kwargs,
):
  if skip_reshape:
    b, _, _, dim_head = q.shape
  else:
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
      lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
      (q, k, v),
    )

  if mask is not None:
    # add a batch dimension if there isn't already one
    if mask.ndim == 2:
      mask = mask.unsqueeze(0)
    # add a heads dimension if there isn't already one
    if mask.ndim == 3:
      mask = mask.unsqueeze(1)

  if SDP_BATCH_LIMIT >= b:
    out = ops.scaled_dot_product_attention(
      q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    if not skip_output_reshape:
      out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
  else:
    out = torch.empty(
      (b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device
    )
    for i in range(0, b, SDP_BATCH_LIMIT):
      m = mask
      if mask is not None:
        if mask.shape[0] > 1:
          m = mask[i : i + SDP_BATCH_LIMIT]

      out[i : i + SDP_BATCH_LIMIT] = (
        ops.scaled_dot_product_attention(
          q[i : i + SDP_BATCH_LIMIT],
          k[i : i + SDP_BATCH_LIMIT],
          v[i : i + SDP_BATCH_LIMIT],
          attn_mask=m,
          dropout_p=0.0,
          is_causal=False,
        )
        .transpose(1, 2)
        .reshape(-1, q.shape[2], heads * dim_head)
      )
  return out


# TODO: switch to flash_attn or sage or ptch
optimized_attention = attention_basic
optimized_attention_masked = optimized_attention


def optimized_attention_for_device(device, mask=False, small_input=False):
  if small_input:
    if model_management.pytorch_attention_enabled():
      return attention_pytorch  # TODO: need to confirm but this is probably slightly faster for small inputs in all cases
    else:
      return attention_basic

  # if device == torch.device("cpu"):
  #   return attention_sub_quad

  if mask:
    return optimized_attention_masked

  return optimized_attention
