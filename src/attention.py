import functools
import torch
from einops import rearrange, repeat, einsum


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


# TODO: switch to flash_attn or sage or ptch
optimized_attention = attention_basic