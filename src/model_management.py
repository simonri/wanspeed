import torch
from enum import Enum
import psutil


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


class VRAMState(Enum):
  DISABLED = 0  # No vram present: no need to move models to vram
  NO_VRAM = 1  # Very low vram: enable all the options to save vram
  LOW_VRAM = 2
  NORMAL_VRAM = 3
  HIGH_VRAM = 4
  SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
  GPU = 0
  CPU = 1
  MPS = 2


cpu_state = CPUState.GPU
vram_state = VRAMState.NORMAL_VRAM


def get_torch_device():
  global directml_enabled
  global cpu_state
  if cpu_state == CPUState.MPS:
    return torch.device("mps")
  if cpu_state == CPUState.CPU:
    return torch.device("cpu")
  else:
    return torch.device(torch.cuda.current_device())


def unet_offload_device():
  if vram_state == VRAMState.HIGH_VRAM:
    return get_torch_device()
  else:
    return torch.device("cpu")


def is_device_cpu(device):
  return is_device_type(device, "cpu")


def cpu_mode():
  global cpu_state
  return cpu_state == CPUState.CPU


def get_total_memory(dev=None, torch_total_too=False):
  global directml_enabled
  if dev is None:
    dev = get_torch_device()

  if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
    mem_total = psutil.virtual_memory().total
    mem_total_torch = mem_total
  else:
    if directml_enabled:
      mem_total = 1024 * 1024 * 1024  # TODO
      mem_total_torch = mem_total
    elif is_intel_xpu():
      stats = torch.xpu.memory_stats(dev)
      mem_reserved = stats["reserved_bytes.all.current"]
      mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
      mem_total_torch = mem_reserved
      mem_total = mem_total_xpu
    else:
      stats = torch.cuda.memory_stats(dev)
      mem_reserved = stats["reserved_bytes.all.current"]
      _, mem_total_cuda = torch.cuda.mem_get_info(dev)
      mem_total_torch = mem_reserved
      mem_total = mem_total_cuda

  if torch_total_too:
    return (mem_total, mem_total_torch)
  else:
    return mem_total


EXTRA_RESERVED_VRAM = 400 * 1024 * 1024


def extra_reserved_memory():
  return EXTRA_RESERVED_VRAM


def minimum_inference_memory():
  return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()


def maximum_vram_for_weights(device=None):
  return get_total_memory(device) * 0.88 - minimum_inference_memory()


try:
  torch_version = torch.version.__version__
  temp = torch_version.split(".")
  torch_version_numeric = (int(temp[0]), int(temp[1]))
except:
  pass


def should_use_bf16(
  device=None, model_params=0, prioritize_performance=True, manual_cast=False
):
  if device is not None:
    if is_device_cpu(device):  # TODO ? bf16 works on CPU but is extremely slow
      return False

  if directml_enabled:
    return False

  if cpu_mode():
    return False

  if is_intel_xpu():
    if torch_version_numeric < (2, 3):
      return True
    else:
      return torch.xpu.is_bf16_supported()

  props = torch.cuda.get_device_properties(device)

  if props.major >= 8:
    return True

  bf16_works = torch.cuda.is_bf16_supported()

  if bf16_works and manual_cast:
    free_model_memory = maximum_vram_for_weights(device)
    if (not prioritize_performance) or model_params * 4 > free_model_memory:
      return True

  return False


def get_supported_float8_types():
  float8_types = []
  try:
    float8_types.append(torch.float8_e4m3fn)
  except:
    pass
  try:
    float8_types.append(torch.float8_e4m3fnuz)
  except:
    pass
  try:
    float8_types.append(torch.float8_e5m2)
  except:
    pass
  try:
    float8_types.append(torch.float8_e5m2fnuz)
  except:
    pass
  try:
    float8_types.append(torch.float8_e8m0fnu)
  except:
    pass
  return float8_types


FLOAT8_TYPES = get_supported_float8_types()


def is_nvidia():
  global cpu_state
  if cpu_state == CPUState.GPU:
    if torch.version.cuda:
      return True
  return False


def supports_fp8_compute(device=None):
  if not is_nvidia():
    return False

  props = torch.cuda.get_device_properties(device)
  if props.major >= 9:
    return True
  if props.major < 8:
    return False
  if props.minor < 9:
    return False

  if torch_version_numeric < (2, 3):
    return False

  return True


def should_use_fp16(
  device=None, model_params=0, prioritize_performance=True, manual_cast=False
):
  if device is not None:
    if is_device_cpu(device):
      return False

  if is_directml_enabled():
    return True

  if device is not None and is_device_mps(device):
    return True

  if cpu_mode():
    return False

  if is_intel_xpu():
    if torch_version_numeric < (2, 3):
      return True
    else:
      return torch.xpu.get_device_properties(device).has_fp16

  if torch.version.hip:
    return True

  props = torch.cuda.get_device_properties(device)
  if props.major >= 8:
    return True

  if props.major < 6:
    return False

  # FP16 is confirmed working on a 1080 (GP104) and on latest pytorch actually seems faster than fp32
  nvidia_10_series = [
    "1080",
    "1070",
    "titan x",
    "p3000",
    "p3200",
    "p4000",
    "p4200",
    "p5000",
    "p5200",
    "p6000",
    "1060",
    "1050",
    "p40",
    "p100",
    "p6",
    "p4",
  ]
  for x in nvidia_10_series:
    if x in props.name.lower():
      if manual_cast:
        return True
      else:
        return False  # weird linux behavior where fp32 is faster

  if manual_cast:
    free_model_memory = maximum_vram_for_weights(device)
    if (not prioritize_performance) or model_params * 4 > free_model_memory:
      return True

  if props.major < 7:
    return False

  # FP16 is just broken on these cards
  nvidia_16_series = [
    "1660",
    "1650",
    "1630",
    "T500",
    "T550",
    "T600",
    "MX550",
    "MX450",
    "CMP 30HX",
    "T2000",
    "T1000",
    "T1200",
  ]
  for x in nvidia_16_series:
    if x in props.name:
      return False

  return True


def unet_dtype(
  device=None,
  model_params=0,
  supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
  weight_dtype=None,
):
  if model_params < 0:
    model_params = 1000000000000000000000

  fp8_dtype = None
  if weight_dtype in FLOAT8_TYPES:
    fp8_dtype = weight_dtype

  if fp8_dtype is not None:
    if supports_fp8_compute(
      device
    ):  # if fp8 compute is supported the casting is most likely not expensive
      return fp8_dtype

    free_model_memory = maximum_vram_for_weights(device)
    if model_params * 2 > free_model_memory:
      return fp8_dtype

  if weight_dtype == torch.float16:
    if torch.float16 in supported_dtypes and should_use_fp16(
      device=device, model_params=model_params
    ):
      return torch.float16

  for dt in supported_dtypes:
    if dt == torch.float16 and should_use_fp16(
      device=device, model_params=model_params
    ):
      if torch.float16 in supported_dtypes:
        return torch.float16
    if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params):
      if torch.bfloat16 in supported_dtypes:
        return torch.bfloat16

  for dt in supported_dtypes:
    if dt == torch.float16 and should_use_fp16(
      device=device, model_params=model_params, manual_cast=True
    ):
      if torch.float16 in supported_dtypes:
        return torch.float16
    if dt == torch.bfloat16 and should_use_bf16(
      device, model_params=model_params, manual_cast=True
    ):
      if torch.bfloat16 in supported_dtypes:
        return torch.bfloat16

  return torch.float32


def unet_manual_cast(
  weight_dtype,
  inference_device,
  supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
  if weight_dtype == torch.float32 or weight_dtype == torch.float64:
    return None

  fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
  if fp16_supported and weight_dtype == torch.float16:
    return None

  bf16_supported = should_use_bf16(inference_device)
  if bf16_supported and weight_dtype == torch.bfloat16:
    return None

  fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)

  for dt in supported_dtypes:
    if dt == torch.float16 and fp16_supported:
      return torch.float16
    if dt == torch.bfloat16 and bf16_supported:
      return torch.bfloat16

  return torch.float32


def module_size(module):
  module_mem = 0
  sd = module.state_dict()
  for k in sd:
    t = sd[k]
    module_mem += t.nelement() * t.element_size()
  return module_mem
