import torch
import src.model_management as model_management
import contextlib
import src.rsnorm as rsnorm
from src.quant_ops import QuantizedTensor


def run_every_op():
  if torch.compiler.is_compiling():
    return

  model_management.throw_exception_if_processing_interrupted()


def cast_bias_weight(
  s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False
):
  # NOTE: offloadable=False is a a legacy and if you are a custom node author reading this please pass
  # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
  # will add async-offload support to your cast and improve performance.
  if input is not None:
    if dtype is None:
      if isinstance(input, QuantizedTensor):
        dtype = input._layout_params["orig_dtype"]
      else:
        dtype = input.dtype
    if bias_dtype is None:
      bias_dtype = dtype
    if device is None:
      device = input.device

  if offloadable and (
    device != s.weight.device or (s.bias is not None and device != s.bias.device)
  ):
    offload_stream = model_management.get_offload_stream(device)
  else:
    offload_stream = None

  if offload_stream is not None:
    wf_context = offload_stream
  else:
    wf_context = contextlib.nullcontext()

  non_blocking = model_management.device_supports_non_blocking(device)

  weight_has_function = len(s.weight_function) > 0
  bias_has_function = len(s.bias_function) > 0

  weight = model_management.cast_to(
    s.weight,
    None,
    device,
    non_blocking=non_blocking,
    copy=weight_has_function,
    stream=offload_stream,
  )

  bias = None
  if s.bias is not None:
    bias = model_management.cast_to(
      s.bias,
      bias_dtype,
      device,
      non_blocking=non_blocking,
      copy=bias_has_function,
      stream=offload_stream,
    )

    if bias_has_function:
      with wf_context:
        for f in s.bias_function:
          bias = f(bias)

  if weight_has_function or weight.dtype != dtype:
    with wf_context:
      weight = weight.to(dtype=dtype)
      for f in s.weight_function:
        weight = f(weight)

  model_management.sync_stream(device, offload_stream)
  if offloadable:
    return weight, bias, offload_stream
  else:
    # Legacy function signature
    return weight, bias


def uncast_bias_weight(s, weight, bias, offload_stream):
  if offload_stream is None:
    return
  if weight is not None:
    device = weight.device
  else:
    if bias is None:
      return
    device = bias.device
  offload_stream.wait_stream(model_management.current_stream(device))


class CastWeightBiasOp:
  comfy_cast_weights = False
  weight_function = []
  bias_function = []


class disable_weight_init:
  class Linear(torch.nn.Linear, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input):
      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = torch.nn.functional.linear(input, weight, bias)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input):
      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = self._conv_forward(input, weight, bias)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input):
      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = self._conv_forward(input, weight, bias)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def _conv_forward(self, input, weight, bias, *args, **kwargs):
      return super()._conv_forward(input, weight, bias, *args, **kwargs)

    def forward_comfy_cast_weights(self, input):
      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = self._conv_forward(input, weight, bias)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input):
      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input):
      if self.weight is not None:
        weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      else:
        weight = None
        bias = None
        offload_stream = None
      x = torch.nn.functional.layer_norm(
        input, self.normalized_shape, weight, bias, self.eps
      )
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class RMSNorm(torch.nn.RMSNorm, CastWeightBiasOp):
    def reset_parameters(self):
      self.bias = None
      return None

    def forward_comfy_cast_weights(self, input):
      if self.weight is not None:
        weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      else:
        weight = None
        bias = None
        offload_stream = None
      x = rsnorm.rms_norm(
        input, weight, self.eps
      )  # TODO: switch to commented out line when old torch is deprecated
      # x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input, output_size=None):
      num_spatial_dims = 2
      output_padding = self._output_padding(
        input,
        output_size,
        self.stride,
        self.padding,
        self.kernel_size,
        num_spatial_dims,
        self.dilation,
      )

      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = torch.nn.functional.conv_transpose2d(
        input,
        weight,
        bias,
        self.stride,
        self.padding,
        output_padding,
        self.groups,
        self.dilation,
      )
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
    def reset_parameters(self):
      return None

    def forward_comfy_cast_weights(self, input, output_size=None):
      num_spatial_dims = 1
      output_padding = self._output_padding(
        input,
        output_size,
        self.stride,
        self.padding,
        self.kernel_size,
        num_spatial_dims,
        self.dilation,
      )

      weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
      x = torch.nn.functional.conv_transpose1d(
        input,
        weight,
        bias,
        self.stride,
        self.padding,
        output_padding,
        self.groups,
        self.dilation,
      )
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        return super().forward(*args, **kwargs)

  class Embedding(torch.nn.Embedding, CastWeightBiasOp):
    def reset_parameters(self):
      self.bias = None
      return None

    def forward_comfy_cast_weights(self, input, out_dtype=None):
      output_dtype = out_dtype
      if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
        out_dtype = None
      weight, bias, offload_stream = cast_bias_weight(
        self, device=input.device, dtype=out_dtype, offloadable=True
      )
      x = torch.nn.functional.embedding(
        input,
        weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
      ).to(dtype=output_dtype)
      uncast_bias_weight(self, weight, bias, offload_stream)
      return x

    def forward(self, *args, **kwargs):
      run_every_op()
      if (
        self.comfy_cast_weights
        or len(self.weight_function) > 0
        or len(self.bias_function) > 0
      ):
        return self.forward_comfy_cast_weights(*args, **kwargs)
      else:
        if "out_dtype" in kwargs:
          kwargs.pop("out_dtype")
        return super().forward(*args, **kwargs)

  @classmethod
  def conv_nd(s, dims, *args, **kwargs):
    if dims == 2:
      return s.Conv2d(*args, **kwargs)
    elif dims == 3:
      return s.Conv3d(*args, **kwargs)
    else:
      raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
  class Linear(disable_weight_init.Linear):
    comfy_cast_weights = True

  class Conv1d(disable_weight_init.Conv1d):
    comfy_cast_weights = True

  class Conv2d(disable_weight_init.Conv2d):
    comfy_cast_weights = True

  class Conv3d(disable_weight_init.Conv3d):
    comfy_cast_weights = True

  class GroupNorm(disable_weight_init.GroupNorm):
    comfy_cast_weights = True

  class LayerNorm(disable_weight_init.LayerNorm):
    comfy_cast_weights = True

  class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
    comfy_cast_weights = True

  class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
    comfy_cast_weights = True

  class RMSNorm(disable_weight_init.RMSNorm):
    comfy_cast_weights = True

  class Embedding(disable_weight_init.Embedding):
    comfy_cast_weights = True


def pick_operations(
  weight_dtype,
  compute_dtype,
  load_device=None,
  disable_fast_fp8=False,
  fp8_optimizations=False,
  scaled_fp8=None,
  model_config=None,
):
  # if (
  #   PerformanceFeature.CublasOps in args.fast
  #   and CUBLAS_IS_AVAILABLE
  #   and weight_dtype == torch.float16
  #   and (compute_dtype == torch.float16 or compute_dtype is None)
  # ):
  #   logging.info("Using cublas ops")
  #   return cublas_ops

  if compute_dtype is None or weight_dtype == compute_dtype:
    return disable_weight_init

  return manual_cast
