import logging
import torch
from enum import Enum
import psutil
import gc
import weakref
import sys


def is_device_type(device, type):
  if hasattr(device, "type"):
    if device.type == type:
      return True
  return False


def is_device_mps(device):
  return is_device_type(device, "mps")


def is_intel_xpu():
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
  if dev is None:
    dev = get_torch_device()

  if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
    mem_total = psutil.virtual_memory().total
    mem_total_torch = mem_total
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


def supports_cast(device, dtype):  # TODO
  if dtype == torch.float32:
    return True
  if dtype == torch.float16:
    return True
  if dtype == torch.bfloat16:
    return True
  if is_device_mps(device):
    return False
  if dtype == torch.float8_e4m3fn:
    return True
  if dtype == torch.float8_e5m2:
    return True
  return False


def text_encoder_initial_device(load_device, offload_device, model_size=0):
  if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
    return offload_device

  if is_device_mps(load_device):
    return load_device

  # TODO: offload?
  return load_device


def text_encoder_offload_device():
  return torch.device("cpu")


def text_encoder_dtype(device=None):
  return torch.float16


def text_encoder_device():
  if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
    if should_use_fp16(prioritize_performance=False):
      return get_torch_device()
    else:
      return torch.device("cpu")
  else:
    return torch.device("cpu")


def soft_empty_cache(force=False):
  global cpu_state
  if cpu_state == CPUState.MPS:
    torch.mps.empty_cache()
  elif is_intel_xpu():
    torch.xpu.empty_cache()
  elif torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


current_loaded_models = []


def cleanup_models_gc():
  do_gc = False
  for i in range(len(current_loaded_models)):
    cur = current_loaded_models[i]
    if cur.is_dead():
      logging.info(
        "Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(
          cur.real_model().__class__.__name__
        )
      )
      do_gc = True
      break

  if do_gc:
    gc.collect()
    soft_empty_cache()

    for i in range(len(current_loaded_models)):
      cur = current_loaded_models[i]
      if cur.is_dead():
        logging.warning(
          "WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(
            cur.real_model().__class__.__name__
          )
        )


def cleanup_models():
  to_delete = []
  for i in range(len(current_loaded_models)):
    if current_loaded_models[i].real_model() is None:
      to_delete = [i] + to_delete

  for i in to_delete:
    x = current_loaded_models.pop(i)
    del x


class LoadedModel:
  def __init__(self, model):
    self._set_model(model)
    self.device = model.load_device
    self.real_model = None
    self.currently_used = True
    self.model_finalizer = None
    self._patcher_finalizer = None

  def _set_model(self, model):
    self._model = weakref.ref(model)
    if model.parent is not None:
      self._parent_model = weakref.ref(model.parent)
      self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

  def _switch_parent(self):
    model = self._parent_model()
    if model is not None:
      self._set_model(model)

  @property
  def model(self):
    return self._model()

  def model_memory(self):
    return self.model.model_size()

  def model_loaded_memory(self):
    return self.model.loaded_size()

  def model_offloaded_memory(self):
    return self.model.model_size() - self.model.loaded_size()

  def model_memory_required(self, device):
    if device == self.model.current_loaded_device():
      return self.model_offloaded_memory()
    else:
      return self.model_memory()

  def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
    self.model.model_patches_to(self.device)
    self.model.model_patches_to(self.model.model_dtype())

    # if self.model.loaded_size() > 0:
    use_more_vram = lowvram_model_memory
    if use_more_vram == 0:
      use_more_vram = 1e32
    self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)

    real_model = self.model.model

    self.real_model = weakref.ref(real_model)
    self.model_finalizer = weakref.finalize(real_model, cleanup_models)
    return real_model

  def should_reload_model(self, force_patch_weights=False):
    if force_patch_weights and self.model.lowvram_patch_counter() > 0:
      return True
    return False

  def model_unload(self, memory_to_free=None, unpatch_weights=True):
    if memory_to_free is not None:
      if memory_to_free < self.model.loaded_size():
        freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
        if freed >= memory_to_free:
          return False
    self.model.detach(unpatch_weights)
    self.model_finalizer.detach()
    self.model_finalizer = None
    self.real_model = None
    return True

  def model_use_more_vram(self, extra_memory, force_patch_weights=False):
    return self.model.partially_load(
      self.device, extra_memory, force_patch_weights=force_patch_weights
    )

  def __eq__(self, other):
    return self.model is other.model

  def __del__(self):
    if self._patcher_finalizer is not None:
      self._patcher_finalizer.detach()

  def is_dead(self):
    return self.real_model() is not None and self.model is None


def get_free_memory(dev=None, torch_free_too=False):
  if dev is None:
    dev = get_torch_device()

  if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
    mem_free_total = psutil.virtual_memory().available
    mem_free_torch = mem_free_total
  else:
    stats = torch.cuda.memory_stats(dev)
    mem_active = stats["active_bytes.all.current"]
    mem_reserved = stats["reserved_bytes.all.current"]
    mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch

  if torch_free_too:
    return (mem_free_total, mem_free_torch)
  else:
    return mem_free_total


def free_memory(memory_required, device, keep_loaded=[]):
  cleanup_models_gc()
  unloaded_model = []
  can_unload = []
  unloaded_models = []

  for i in range(len(current_loaded_models) - 1, -1, -1):
    shift_model = current_loaded_models[i]
    if shift_model.device == device:
      if shift_model not in keep_loaded and not shift_model.is_dead():
        can_unload.append(
          (
            -shift_model.model_offloaded_memory(),
            sys.getrefcount(shift_model.model),
            shift_model.model_memory(),
            i,
          )
        )
        shift_model.currently_used = False

  for x in sorted(can_unload):
    i = x[-1]
    memory_to_free = None
    free_mem = get_free_memory(device)
    if free_mem > memory_required:
      break
    memory_to_free = memory_required - free_mem
    logging.debug(
      f"Unloading {current_loaded_models[i].model.model.__class__.__name__}"
    )
    if current_loaded_models[i].model_unload(memory_to_free):
      unloaded_model.append(i)

  for i in sorted(unloaded_model, reverse=True):
    unloaded_models.append(current_loaded_models.pop(i))

  if len(unloaded_model) > 0:
    soft_empty_cache()
  else:
    if vram_state != VRAMState.HIGH_VRAM:
      mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
      if mem_free_torch > mem_free_total * 0.25:
        soft_empty_cache()
  return unloaded_models


def load_models_gpu(
  models,
  memory_required=0,
  force_patch_weights=False,
  minimum_memory_required=None,
  force_full_load=False,
):
  cleanup_models_gc()
  global vram_state

  inference_memory = minimum_inference_memory()
  extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
  if minimum_memory_required is None:
    minimum_memory_required = extra_mem
  else:
    minimum_memory_required = max(
      inference_memory, minimum_memory_required + extra_reserved_memory()
    )

  models_temp = set()
  for m in models:
    models_temp.add(m)
    for mm in m.model_patches_models():
      models_temp.add(mm)

  models = models_temp

  models_to_load = []

  for x in models:
    loaded_model = LoadedModel(x)
    try:
      loaded_model_index = current_loaded_models.index(loaded_model)
    except:
      loaded_model_index = None

    if loaded_model_index is not None:
      loaded = current_loaded_models[loaded_model_index]
      loaded.currently_used = True
      models_to_load.append(loaded)
    else:
      if hasattr(x, "model"):
        logging.info(f"Requested to load {x.model.__class__.__name__}")
      models_to_load.append(loaded_model)

  for loaded_model in models_to_load:
    to_unload = []
    for i in range(len(current_loaded_models)):
      if loaded_model.model.is_clone(current_loaded_models[i].model):
        to_unload = [i] + to_unload
    for i in to_unload:
      model_to_unload = current_loaded_models.pop(i)
      model_to_unload.model.detach(unpatch_all=False)
      model_to_unload.model_finalizer.detach()

  total_memory_required = {}
  for loaded_model in models_to_load:
    total_memory_required[loaded_model.device] = total_memory_required.get(
      loaded_model.device, 0
    ) + loaded_model.model_memory_required(loaded_model.device)

  for device in total_memory_required:
    if device != torch.device("cpu"):
      free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

  for device in total_memory_required:
    if device != torch.device("cpu"):
      free_mem = get_free_memory(device)
      if free_mem < minimum_memory_required:
        models_l = free_memory(minimum_memory_required, device)
        logging.info("{} models unloaded.".format(len(models_l)))

  for loaded_model in models_to_load:
    model = loaded_model.model
    torch_dev = model.load_device
    if is_device_cpu(torch_dev):
      vram_set_state = VRAMState.DISABLED
    else:
      vram_set_state = vram_state
    lowvram_model_memory = 0
    if (
      vram_set_state == VRAMState.LOW_VRAM
      or vram_set_state == VRAMState.NORMAL_VRAM
      and not force_full_load
    ):
      loaded_memory = loaded_model.model_loaded_memory()
      current_free_mem = get_free_memory(torch_dev) + loaded_memory

      lowvram_model_memory = max(
        128 * 1024 * 1024,
        (current_free_mem - minimum_memory_required),
        min(
          current_free_mem * 0,
          current_free_mem - minimum_inference_memory(),
        ),
      )
      lowvram_model_memory = lowvram_model_memory - loaded_memory

      if lowvram_model_memory == 0:
        lowvram_model_memory = 0.1

    if vram_set_state == VRAMState.NO_VRAM:
      lowvram_model_memory = 0.1

    loaded_model.model_load(
      lowvram_model_memory, force_patch_weights=force_patch_weights
    )
    current_loaded_models.insert(0, loaded_model)
  return
