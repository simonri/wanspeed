import torch
import src.model_management as model_management

rms_norm_torch = torch.nn.functional.rms_norm


def rms_norm(x, weight=None, eps=1e-6):
  if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
    if weight is None:
      return rms_norm_torch(x, (x.shape[-1],), eps=eps)
    else:
      return rms_norm_torch(
        x,
        weight.shape,
        weight=model_management.cast_to(weight, dtype=x.dtype, device=x.device),
        eps=eps,
      )
  else:
    r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    if weight is None:
      return r
    else:
      return r * model_management.cast_to(weight, dtype=x.dtype, device=x.device)
