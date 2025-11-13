import torch
import logging

_LAYOUT_REGISTRY = {}
_GENERIC_UTILS = {}


def _get_layout_from_args(args):
  for arg in args:
    if isinstance(arg, QuantizedTensor):
      return arg._layout_type
    elif isinstance(arg, (list, tuple)):
      for item in arg:
        if isinstance(item, QuantizedTensor):
          return item._layout_type
  return None


class QuantizedLayout:
  """
  Base class for quantization layouts.

  A layout encapsulates the format-specific logic for quantization/dequantization
  and provides a uniform interface for extracting raw tensors needed for computation.

  New quantization formats should subclass this and implement the required methods.
  """

  @classmethod
  def quantize(cls, tensor, **kwargs) -> tuple[torch.Tensor, dict]:
    raise NotImplementedError(f"{cls.__name__} must implement quantize()")

  @staticmethod
  def dequantize(qdata, **layout_params) -> torch.Tensor:
    raise NotImplementedError("TensorLayout must implement dequantize()")

  @classmethod
  def get_plain_tensors(cls, qtensor) -> torch.Tensor:
    raise NotImplementedError(f"{cls.__name__} must implement get_plain_tensors()")


class TensorCoreFP8Layout(QuantizedLayout):
  """
  Storage format:
  - qdata: FP8 tensor (torch.float8_e4m3fn or torch.float8_e5m2)
  - scale: Scalar tensor (float32) for dequantization
  - orig_dtype: Original dtype before quantization (for casting back)
  """

  @classmethod
  def quantize(cls, tensor, scale=None, dtype=torch.float8_e4m3fn):
    orig_dtype = tensor.dtype

    if scale is None:
      scale = torch.amax(tensor.abs()) / torch.finfo(dtype).max

    if not isinstance(scale, torch.Tensor):
      scale = torch.tensor(scale)
    scale = scale.to(device=tensor.device, dtype=torch.float32)

    tensor_scaled = tensor * (1.0 / scale).to(tensor.dtype)
    # TODO: uncomment this if it's actually needed because the clamp has a small performance penality'
    # lp_amax = torch.finfo(dtype).max
    # torch.clamp(tensor_scaled, min=-lp_amax, max=lp_amax, out=tensor_scaled)
    qdata = tensor_scaled.to(dtype, memory_format=torch.contiguous_format)

    layout_params = {"scale": scale, "orig_dtype": orig_dtype}
    return qdata, layout_params

  @staticmethod
  def dequantize(qdata, scale, orig_dtype, **kwargs):
    plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
    return plain_tensor * scale

  @classmethod
  def get_plain_tensors(cls, qtensor):
    return qtensor._qdata, qtensor._layout_params["scale"]


LAYOUTS = {
  "TensorCoreFP8Layout": TensorCoreFP8Layout,
}


class QuantizedTensor(torch.Tensor):
  """
  Universal quantized tensor that works with any layout.

  This tensor subclass uses a pluggable layout system to support multiple
  quantization formats (FP8, INT4, INT8, etc.) without code duplication.

  The layout_type determines format-specific behavior, while common operations
  (detach, clone, to) are handled generically.

  Attributes:
      _qdata: The quantized tensor data
      _layout_type: Layout class (e.g., TensorCoreFP8Layout)
      _layout_params: Dict with layout-specific params (scale, zero_point, etc.)
  """

  @staticmethod
  def __new__(cls, qdata, layout_type, layout_params):
    """
    Create a quantized tensor.

    Args:
        qdata: The quantized data tensor
        layout_type: Layout class (subclass of QuantizedLayout)
        layout_params: Dict with layout-specific parameters
    """
    return torch.Tensor._make_wrapper_subclass(
      cls, qdata.shape, device=qdata.device, dtype=qdata.dtype, requires_grad=False
    )

  def __init__(self, qdata, layout_type, layout_params):
    self._qdata = qdata
    self._layout_type = layout_type
    self._layout_params = layout_params

  def __repr__(self):
    layout_name = self._layout_type
    param_str = ", ".join(f"{k}={v}" for k, v in list(self._layout_params.items())[:2])
    return f"QuantizedTensor(shape={self.shape}, layout={layout_name}, {param_str})"

  @property
  def layout_type(self):
    return self._layout_type

  def __tensor_flatten__(self):
    """
    Tensor flattening protocol for proper device movement.
    """
    inner_tensors = ["_qdata"]
    ctx = {
      "layout_type": self._layout_type,
    }

    tensor_params = {}
    non_tensor_params = {}
    for k, v in self._layout_params.items():
      if isinstance(v, torch.Tensor):
        tensor_params[k] = v
      else:
        non_tensor_params[k] = v

    ctx["tensor_param_keys"] = list(tensor_params.keys())
    ctx["non_tensor_params"] = non_tensor_params

    for k, v in tensor_params.items():
      attr_name = f"_layout_param_{k}"
      object.__setattr__(self, attr_name, v)
      inner_tensors.append(attr_name)

    return inner_tensors, ctx

  @staticmethod
  def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
    """
    Tensor unflattening protocol for proper device movement.
    Reconstructs the QuantizedTensor after device movement.
    """
    layout_type = ctx["layout_type"]
    layout_params = dict(ctx["non_tensor_params"])

    for key in ctx["tensor_param_keys"]:
      attr_name = f"_layout_param_{key}"
      layout_params[key] = inner_tensors[attr_name]

    return QuantizedTensor(inner_tensors["_qdata"], layout_type, layout_params)

  @classmethod
  def from_float(cls, tensor, layout_type, **quantize_kwargs) -> "QuantizedTensor":
    qdata, layout_params = LAYOUTS[layout_type].quantize(tensor, **quantize_kwargs)
    return cls(qdata, layout_type, layout_params)

  def dequantize(self) -> torch.Tensor:
    return LAYOUTS[self._layout_type].dequantize(self._qdata, **self._layout_params)

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}

    # Step 1: Check generic utilities first (detach, clone, to, etc.)
    if func in _GENERIC_UTILS:
      return _GENERIC_UTILS[func](func, args, kwargs)

    # Step 2: Check layout-specific handlers (linear, matmul, etc.)
    layout_type = _get_layout_from_args(args)
    if layout_type and func in _LAYOUT_REGISTRY:
      handler = _LAYOUT_REGISTRY[func].get(layout_type)
      if handler:
        return handler(func, args, kwargs)

    # Step 3: Fallback to dequantization
    if isinstance(args[0] if args else None, QuantizedTensor):
      logging.info(
        f"QuantizedTensor: Unhandled operation {func}, falling back to dequantization. kwargs={kwargs}"
      )
    return cls._dequant_and_fallback(func, args, kwargs)

  @classmethod
  def _dequant_and_fallback(cls, func, args, kwargs):
    def dequant_arg(arg):
      if isinstance(arg, QuantizedTensor):
        return arg.dequantize()
      elif isinstance(arg, (list, tuple)):
        return type(arg)(dequant_arg(a) for a in arg)
      return arg

    new_args = dequant_arg(args)
    new_kwargs = dequant_arg(kwargs)
    return func(*new_args, **new_kwargs)
