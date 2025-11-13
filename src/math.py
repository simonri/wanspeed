from torch import Tensor

def apply_rope1(x: Tensor, freqs_cis: Tensor):
  x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)

  x_out = freqs_cis[..., 0] * x_[..., 0]
  x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])

  return x_out.reshape(*x.shape).type_as(x)
