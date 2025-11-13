import torch
import logging
import math
from enum import Enum
from src.wan.model import WanModel
from src.model_sampling import model_sampling
from src.ops import pick_operations
from src.model_management import cast_to_device
from src.conds import CONDRegular, CONDNoiseShape, CONDCrossAttn
import src.utils as utils
import src.model_management as model_management
import src.patcher_extension as patcher_extension


class ModelType(Enum):
  EPS = 1
  V_PREDICTION = 2
  V_PREDICTION_EDM = 3
  STABLE_CASCADE = 4
  EDM = 5
  FLOW = 6
  V_PREDICTION_CONTINUOUS = 7
  FLUX = 8
  IMG_TO_IMG = 9
  FLOW_COSMOS = 10


def convert_tensor(extra, dtype, device):
  if hasattr(extra, "dtype"):
    if extra.dtype != torch.int and extra.dtype != torch.long:
      extra = cast_to_device(extra, device, dtype)
    else:
      extra = cast_to_device(extra, device, None)
  return extra


class BaseModel(torch.nn.Module):
  def __init__(
    self, model_config, model_type=ModelType.EPS, device=None, unet_model=None
  ):
    super().__init__()

    unet_config = model_config.unet_config
    self.latent_format = model_config.latent_format
    self.model_config = model_config
    self.manual_cast_dtype = model_config.manual_cast_dtype
    self.device = device
    self.current_patcher = None

    if not unet_config.get("disable_unet_model_creation", False):
      if model_config.custom_operations is None:
        fp8 = model_config.optimizations.get("fp8", False)
        operations = pick_operations(
          unet_config.get("dtype", None),
          self.manual_cast_dtype,
          fp8_optimizations=fp8,
          scaled_fp8=model_config.scaled_fp8,
          model_config=model_config,
        )
      else:
        operations = model_config.custom_operations
      self.diffusion_model = unet_model(
        **unet_config, device=device, operations=operations
      )
      self.diffusion_model.eval()
      # if comfy.model_management.force_channels_last():
      #   self.diffusion_model.to(memory_format=torch.channels_last)
      #   logging.debug("using channels last mode for diffusion model")
      logging.info(
        "model weight dtype {}, manual cast: {}".format(
          self.get_dtype(), self.manual_cast_dtype
        )
      )
    self.model_type = model_type
    self.model_sampling = model_sampling(model_config, model_type)

    self.adm_channels = unet_config.get("adm_in_channels", None)
    if self.adm_channels is None:
      self.adm_channels = 0

    self.concat_keys = ()
    logging.info("model_type {}".format(model_type.name))
    logging.debug("adm {}".format(self.adm_channels))
    self.memory_usage_factor = model_config.memory_usage_factor
    self.memory_usage_factor_conds = ()
    self.memory_usage_shape_process = {}

  def apply_model(
    self,
    x,
    t,
    c_concat=None,
    c_crossattn=None,
    control=None,
    transformer_options={},
    **kwargs,
  ):
    return patcher_extension.WrapperExecutor.new_class_executor(
      self._apply_model,
      self,
      patcher_extension.get_all_wrappers(
        patcher_extension.WrappersMP.APPLY_MODEL, transformer_options
      ),
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

  def _apply_model(
    self,
    x,
    t,
    c_concat=None,
    c_crossattn=None,
    control=None,
    transformer_options={},
    **kwargs,
  ):
    sigma = t
    xc = self.model_sampling.calculate_input(sigma, x)

    if c_concat is not None:
      xc = torch.cat(
        [xc] + [cast_to_device(c_concat, xc.device, xc.dtype)],
        dim=1,
      )

    context = c_crossattn
    dtype = self.get_dtype()

    if self.manual_cast_dtype is not None:
      dtype = self.manual_cast_dtype

    xc = xc.to(dtype)
    device = xc.device
    t = self.model_sampling.timestep(t).float()
    if context is not None:
      context = cast_to_device(context, device, dtype)

    extra_conds = {}
    for o in kwargs:
      extra = kwargs[o]

      if hasattr(extra, "dtype"):
        extra = convert_tensor(extra, dtype, device)
      elif isinstance(extra, list):
        ex = []
        for ext in extra:
          ex.append(convert_tensor(ext, dtype, device))
        extra = ex
      extra_conds[o] = extra

    t = self.process_timestep(t, x=x, **extra_conds)
    if "latent_shapes" in extra_conds:
      xc = utils.unpack_latents(xc, extra_conds.pop("latent_shapes"))

    model_output = self.diffusion_model(
      xc,
      t,
      context=context,
      control=control,
      transformer_options=transformer_options,
      **extra_conds,
    )
    if len(model_output) > 1 and not torch.is_tensor(model_output):
      model_output, _ = utils.pack_latents(model_output)

    return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

  def process_timestep(self, timestep, **kwargs):
    return timestep

  def get_dtype(self):
    return self.diffusion_model.dtype

  def encode_adm(self, **kwargs):
    return None

  def concat_cond(self, **kwargs):
    if len(self.concat_keys) > 0:
      cond_concat = []
      denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
      concat_latent_image = kwargs.get("concat_latent_image", None)
      if concat_latent_image is None:
        concat_latent_image = kwargs.get("latent_image", None)
      else:
        concat_latent_image = self.process_latent_in(concat_latent_image)

      noise = kwargs.get("noise", None)
      device = kwargs["device"]

      if concat_latent_image.shape[1:] != noise.shape[1:]:
        concat_latent_image = utils.common_upscale(
          concat_latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center"
        )
        if noise.ndim == 5:
          if concat_latent_image.shape[-3] < noise.shape[-3]:
            concat_latent_image = torch.nn.functional.pad(
              concat_latent_image,
              (0, 0, 0, 0, 0, noise.shape[-3] - concat_latent_image.shape[-3]),
              "constant",
              0,
            )
          else:
            concat_latent_image = concat_latent_image[:, :, : noise.shape[-3]]

      concat_latent_image = utils.resize_to_batch_size(
        concat_latent_image, noise.shape[0]
      )

      if denoise_mask is not None:
        if len(denoise_mask.shape) == len(noise.shape):
          denoise_mask = denoise_mask[:, :1]

        num_dim = noise.ndim - 2
        denoise_mask = denoise_mask.reshape(
          (-1, 1) + tuple(denoise_mask.shape[-num_dim:])
        )
        if denoise_mask.shape[-2:] != noise.shape[-2:]:
          denoise_mask = utils.common_upscale(
            denoise_mask, noise.shape[-1], noise.shape[-2], "bilinear", "center"
          )
        denoise_mask = utils.resize_to_batch_size(denoise_mask.round(), noise.shape[0])

      for ck in self.concat_keys:
        if denoise_mask is not None:
          if ck == "mask":
            cond_concat.append(denoise_mask.to(device))
          elif ck == "masked_image":
            cond_concat.append(
              concat_latent_image.to(device)
            )  # NOTE: the latent_image should be masked by the mask in pixel space
          elif ck == "mask_inverted":
            cond_concat.append(1.0 - denoise_mask.to(device))
        else:
          if ck == "mask":
            cond_concat.append(torch.ones_like(noise)[:, :1])
          elif ck == "masked_image":
            cond_concat.append(self.blank_inpaint_image_like(noise))
          elif ck == "mask_inverted":
            cond_concat.append(torch.zeros_like(noise)[:, :1])
        if ck == "concat_image":
          if concat_latent_image is not None:
            cond_concat.append(concat_latent_image.to(device))
          else:
            cond_concat.append(torch.zeros_like(noise))
      data = torch.cat(cond_concat, dim=1)
      return data
    return None

  def extra_conds(self, **kwargs):
    out = {}
    concat_cond = self.concat_cond(**kwargs)
    if concat_cond is not None:
      out["c_concat"] = CONDNoiseShape(concat_cond)

    adm = self.encode_adm(**kwargs)
    if adm is not None:
      out["y"] = CONDRegular(adm)

    cross_attn = kwargs.get("cross_attn", None)
    if cross_attn is not None:
      out["c_crossattn"] = CONDCrossAttn(cross_attn)

    cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
    if cross_attn_cnet is not None:
      out["crossattn_controlnet"] = CONDCrossAttn(cross_attn_cnet)

    c_concat = kwargs.get("noise_concat", None)
    if c_concat is not None:
      out["c_concat"] = CONDNoiseShape(c_concat)

    return out

  def load_model_weights(self, sd, unet_prefix=""):
    to_load = {}
    keys = list(sd.keys())
    for k in keys:
      if k.startswith(unet_prefix):
        to_load[k[len(unet_prefix) :]] = sd.pop(k)

    to_load = self.model_config.process_unet_state_dict(to_load)
    m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
    if len(m) > 0:
      logging.warning("unet missing: {}".format(m))

    if len(u) > 0:
      logging.warning("unet unexpected: {}".format(u))
    del to_load
    return self

  def process_latent_in(self, latent):
    return self.latent_format.process_in(latent)

  def process_latent_out(self, latent):
    return self.latent_format.process_out(latent)

  def state_dict_for_saving(
    self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None
  ):
    extra_sds = []
    if clip_state_dict is not None:
      extra_sds.append(
        self.model_config.process_clip_state_dict_for_saving(clip_state_dict)
      )
    if vae_state_dict is not None:
      extra_sds.append(
        self.model_config.process_vae_state_dict_for_saving(vae_state_dict)
      )
    if clip_vision_state_dict is not None:
      extra_sds.append(
        self.model_config.process_clip_vision_state_dict_for_saving(
          clip_vision_state_dict
        )
      )

    unet_state_dict = self.diffusion_model.state_dict()

    if self.model_config.scaled_fp8 is not None:
      unet_state_dict["scaled_fp8"] = torch.tensor(
        [], dtype=self.model_config.scaled_fp8
      )

    # Save mixed precision metadata
    if (
      hasattr(self.model_config, "layer_quant_config")
      and self.model_config.layer_quant_config
    ):
      metadata = {
        "format_version": "1.0",
        "layers": self.model_config.layer_quant_config,
      }
      unet_state_dict["_quantization_metadata"] = metadata

    unet_state_dict = self.model_config.process_unet_state_dict_for_saving(
      unet_state_dict
    )

    if self.model_type == ModelType.V_PREDICTION:
      unet_state_dict["v_pred"] = torch.tensor([])

    for sd in extra_sds:
      unet_state_dict.update(sd)

    return unet_state_dict

  def set_inpaint(self):
    self.concat_keys = ("mask", "masked_image")

    def blank_inpaint_image_like(latent_image):
      blank_image = torch.ones_like(latent_image)
      # these are the values for "zero" in pixel space translated to latent space
      blank_image[:, 0] *= 0.8223
      blank_image[:, 1] *= -0.6876
      blank_image[:, 2] *= 0.6364
      blank_image[:, 3] *= 0.1380
      return blank_image

    self.blank_inpaint_image_like = blank_inpaint_image_like

  def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
    return self.model_sampling.noise_scaling(
      sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)),
      noise,
      latent_image,
    )

  def memory_required(self, input_shape, cond_shapes={}):
    input_shapes = [input_shape]
    for c in self.memory_usage_factor_conds:
      shape = cond_shapes.get(c, None)
      if shape is not None:
        if c in self.memory_usage_shape_process:
          out = []
          for s in shape:
            out.append(self.memory_usage_shape_process[c](s))
          shape = out

        if len(shape) > 0:
          input_shapes += shape

    if model_management.pytorch_attention_flash_attention():
      dtype = self.get_dtype()
      if self.manual_cast_dtype is not None:
        dtype = self.manual_cast_dtype
      # TODO: this needs to be tweaked
      area = sum(
        map(
          lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes
        )
      )
      return (
        area * model_management.dtype_size(dtype) * 0.01 * self.memory_usage_factor
      ) * (1024 * 1024)
    else:
      # TODO: this formula might be too aggressive since I tweaked the sub-quad and split algorithms to use less memory.
      area = sum(
        map(
          lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes
        )
      )
      return (area * 0.15 * self.memory_usage_factor) * (1024 * 1024)

  def extra_conds_shapes(self, **kwargs):
    return {}


class WAN21(BaseModel):
  def __init__(
    self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None
  ):
    super().__init__(model_config, model_type, device=device, unet_model=WanModel)
    self.image_to_video = image_to_video

  def concat_cond(self, **kwargs):
    noise = kwargs.get("noise", None)
    extra_channels = (
      self.diffusion_model.patch_embedding.weight.shape[1] - noise.shape[1]
    )
    if extra_channels == 0:
      return None

    image = kwargs.get("concat_latent_image", None)
    device = kwargs["device"]

    if image is None:
      shape_image = list(noise.shape)
      shape_image[1] = extra_channels
      image = torch.zeros(
        shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device
      )
    else:
      latent_dim = self.latent_format.latent_channels
      image = utils.common_upscale(
        image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center"
      )
      for i in range(0, image.shape[1], latent_dim):
        image[:, i : i + latent_dim] = self.process_latent_in(
          image[:, i : i + latent_dim]
        )
      image = utils.resize_to_batch_size(image, noise.shape[0])

    if extra_channels != image.shape[1] + 4:
      if not self.image_to_video or extra_channels == image.shape[1]:
        return image

    if image.shape[1] > (extra_channels - 4):
      image = image[:, : (extra_channels - 4)]

    mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
    if mask is None:
      mask = torch.zeros_like(noise)[:, :4]
    else:
      if mask.shape[1] != 4:
        mask = torch.mean(mask, dim=1, keepdim=True)
      mask = 1.0 - mask
      mask = utils.common_upscale(
        mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center"
      )
      if mask.shape[-3] < noise.shape[-3]:
        mask = torch.nn.functional.pad(
          mask,
          (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]),
          mode="constant",
          value=0,
        )
      if mask.shape[1] == 1:
        mask = mask.repeat(1, 4, 1, 1, 1)
      mask = utils.resize_to_batch_size(mask, noise.shape[0])

    concat_mask_index = kwargs.get("concat_mask_index", 0)
    if concat_mask_index != 0:
      return torch.cat(
        (image[:, :concat_mask_index], mask, image[:, concat_mask_index:]), dim=1
      )
    else:
      return torch.cat((mask, image), dim=1)

  def extra_conds(self, **kwargs):
    out = super().extra_conds(**kwargs)
    cross_attn = kwargs.get("cross_attn", None)
    if cross_attn is not None:
      out["c_crossattn"] = CONDRegular(cross_attn)

    clip_vision_output = kwargs.get("clip_vision_output", None)
    if clip_vision_output is not None:
      out["clip_fea"] = CONDRegular(clip_vision_output.penultimate_hidden_states)

    time_dim_concat = kwargs.get("time_dim_concat", None)
    if time_dim_concat is not None:
      out["time_dim_concat"] = CONDRegular(self.process_latent_in(time_dim_concat))

    reference_latents = kwargs.get("reference_latents", None)
    if reference_latents is not None:
      out["reference_latent"] = CONDRegular(
        self.process_latent_in(reference_latents[-1])[:, :, 0]
      )

    return out


class WAN22(WAN21):
  def __init__(
    self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None
  ):
    super(WAN21, self).__init__(
      model_config, model_type, device=device, unet_model=WanModel
    )
    self.image_to_video = image_to_video

  def extra_conds(self, **kwargs):
    out = super().extra_conds(**kwargs)
    denoise_mask = kwargs.get("denoise_mask", None)
    if denoise_mask is not None:
      out["denoise_mask"] = CONDRegular(denoise_mask)
    return out

  def process_timestep(self, timestep, x, denoise_mask=None, **kwargs):
    if denoise_mask is None:
      return timestep
    temp_ts = (
      torch.mean(denoise_mask[:, :, :, :, :], dim=(1, 3, 4), keepdim=True)
      * timestep.view([timestep.shape[0]] + [1] * (denoise_mask.ndim - 1))
    ).reshape(timestep.shape[0], -1)
    return temp_ts

  def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
    return latent_image
