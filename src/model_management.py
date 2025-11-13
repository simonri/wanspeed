import torch


def is_device_type(device, type):
  if hasattr(device, "type"):
    if device.type == type:
      return True
  return False


def is_device_mps(device):
  return is_device_type(device, "mps")


def is_intel_xpu():
  return False


def is_directml_enabled():
  global directml_enabled
  if directml_enabled:
    return True

  return False


def cast_to(
  weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None
):
  if device is None or weight.device == device:
    if not copy:
      if dtype is None or weight.dtype == dtype:
        return weight
    if stream is not None:
      with stream:
        return weight.to(dtype=dtype, copy=copy)
    return weight.to(dtype=dtype, copy=copy)

  if stream is not None:
    with stream:
      r = torch.empty_like(weight, dtype=dtype, device=device)
      r.copy_(weight, non_blocking=non_blocking)
  else:
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
  return r


def device_supports_non_blocking(device):
  if is_device_mps(device):
    return False  # pytorch bug? mps doesn't support non blocking
  if is_intel_xpu():  # xpu does support non blocking but it is slower on iGPUs for some reason so disable by default until situation changes
    return False
  if directml_enabled:
    return False
  return True


def cast_to_device(tensor, device, dtype, copy=False):
  non_blocking = device_supports_non_blocking(device)
  return cast_to(
    tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy
  )


def dtype_size(dtype):
  dtype_size = 4
  if dtype == torch.float16 or dtype == torch.bfloat16:
    dtype_size = 2
  elif dtype == torch.float32:
    dtype_size = 4
  else:
    try:
      dtype_size = dtype.itemsize
    except:  # Old pytorch doesn't have .itemsize
      pass
  return dtype_size


def pytorch_attention_flash_attention():
  # TODO: implement this
  return False

def pytorch_attention_enabled():
  # TODO: implement this we want it
  return True