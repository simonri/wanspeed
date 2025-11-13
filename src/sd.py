import logging
import os
import torch
from enum import Enum
import src.model_management as model_management
import src.utils as utils
import src.model_patcher as model_patcher
import src.model_detection as model_detection
import src.text_encoders as text_encoders
import src.hooks as hooks


class TEModel(Enum):
  CLIP_L = 1
  CLIP_H = 2
  CLIP_G = 3
  T5_XXL = 4
  T5_XL = 5
  T5_BASE = 6
  LLAMA3_8 = 7
  T5_XXL_OLD = 8
  GEMMA_2_2B = 9
  QWEN25_3B = 10
  QWEN25_7B = 11
  BYT5_SMALL_GLYPH = 12
  GEMMA_3_4B = 13


def detect_te_model(sd):
  if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
    return TEModel.CLIP_G
  if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
    return TEModel.CLIP_H
  if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
    return TEModel.CLIP_L
  if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
    weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
    if weight.shape[-1] == 4096:
      return TEModel.T5_XXL
    elif weight.shape[-1] == 2048:
      return TEModel.T5_XL
  if "encoder.block.23.layer.1.DenseReluDense.wi.weight" in sd:
    return TEModel.T5_XXL_OLD
  if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
    weight = sd["encoder.block.0.layer.0.SelfAttention.k.weight"]
    if weight.shape[0] == 384:
      return TEModel.BYT5_SMALL_GLYPH
    return TEModel.T5_BASE
  if "model.layers.0.post_feedforward_layernorm.weight" in sd:
    if "model.layers.0.self_attn.q_norm.weight" in sd:
      return TEModel.GEMMA_3_4B
    return TEModel.GEMMA_2_2B
  if "model.layers.0.self_attn.k_proj.bias" in sd:
    weight = sd["model.layers.0.self_attn.k_proj.bias"]
    if weight.shape[0] == 256:
      return TEModel.QWEN25_3B
    if weight.shape[0] == 512:
      return TEModel.QWEN25_7B
  if "model.layers.0.post_attention_layernorm.weight" in sd:
    return TEModel.LLAMA3_8
  return None


def t5xxl_detect(clip_data):
  weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
  weight_name_old = "encoder.block.23.layer.1.DenseReluDense.wi.weight"

  for sd in clip_data:
    if weight_name in sd or weight_name_old in sd:
      return text_encoders.sd3_clip.t5_xxl_detect(sd)

  return {}


class CLIP:
  def __init__(
    self,
    target=None,
    embedding_directory=None,
    no_init=False,
    tokenizer_data={},
    parameters=0,
    model_options={},
  ):
    if no_init:
      return
    params = target.params.copy()
    clip = target.clip
    tokenizer = target.tokenizer

    load_device = model_options.get(
      "load_device", model_management.text_encoder_device()
    )
    offload_device = model_options.get(
      "offload_device", model_management.text_encoder_offload_device()
    )
    dtype = model_options.get("dtype", None)
    if dtype is None:
      dtype = model_management.text_encoder_dtype(load_device)

    params["dtype"] = dtype
    params["device"] = model_options.get(
      "initial_device",
      model_management.text_encoder_initial_device(
        load_device, offload_device, parameters * model_management.dtype_size(dtype)
      ),
    )
    params["model_options"] = model_options

    self.cond_stage_model = clip(**(params))

    for dt in self.cond_stage_model.dtypes:
      if not model_management.supports_cast(load_device, dt):
        load_device = offload_device
        if params["device"] != offload_device:
          self.cond_stage_model.to(offload_device)
          logging.warning("Had to shift TE back.")

    self.tokenizer = tokenizer(
      embedding_directory=embedding_directory, tokenizer_data=tokenizer_data
    )
    self.patcher = model_patcher.ModelPatcher(
      self.cond_stage_model, load_device=load_device, offload_device=offload_device
    )
    self.patcher.hook_mode = hooks.EnumHookMode.MinVram
    self.patcher.is_clip = True
    self.apply_hooks_to_conds = None
    if params["device"] == load_device:
      model_management.load_models_gpu([self.patcher], force_full_load=True)
    self.layer_idx = None
    self.use_clip_schedule = False
    logging.info(
      "CLIP/text encoder model load device: {}, offload device: {}, current: {}, dtype: {}".format(
        load_device, offload_device, params["device"], dtype
      )
    )
    self.tokenizer_options = {}

  def clone(self):
    n = CLIP(no_init=True)
    n.patcher = self.patcher.clone()
    n.cond_stage_model = self.cond_stage_model
    n.tokenizer = self.tokenizer
    n.layer_idx = self.layer_idx
    n.tokenizer_options = self.tokenizer_options.copy()
    n.use_clip_schedule = self.use_clip_schedule
    n.apply_hooks_to_conds = self.apply_hooks_to_conds
    return n

  def get_ram_usage(self):
    return self.patcher.get_ram_usage()

  def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
    return self.patcher.add_patches(patches, strength_patch, strength_model)

  def set_tokenizer_option(self, option_name, value):
    self.tokenizer_options[option_name] = value

  def clip_layer(self, layer_idx):
    self.layer_idx = layer_idx

  def tokenize(self, text, return_word_ids=False, **kwargs):
    tokenizer_options = kwargs.get("tokenizer_options", {})
    if len(self.tokenizer_options) > 0:
      tokenizer_options = {**self.tokenizer_options, **tokenizer_options}
    if len(tokenizer_options) > 0:
      kwargs["tokenizer_options"] = tokenizer_options
    return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

  def add_hooks_to_dict(self, pooled_dict: dict[str]):
    if self.apply_hooks_to_conds:
      pooled_dict["hooks"] = self.apply_hooks_to_conds
    return pooled_dict

  def encode_from_tokens_scheduled(
    self, tokens, unprojected=False, add_dict: dict[str] = {}, show_pbar=False
  ):
    all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
    all_hooks = self.patcher.forced_hooks
    if all_hooks is None or not self.use_clip_schedule:
      # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
      return_pooled = "unprojected" if unprojected else True
      pooled_dict = self.encode_from_tokens(
        tokens, return_pooled=return_pooled, return_dict=True
      )
      cond = pooled_dict.pop("cond")
      # add/update any keys with the provided add_dict
      pooled_dict.update(add_dict)
      all_cond_pooled.append([cond, pooled_dict])
    else:
      scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

      self.cond_stage_model.reset_clip_options()
      if self.layer_idx is not None:
        self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
      if unprojected:
        self.cond_stage_model.set_clip_options({"projected_pooled": False})

      self.load_model()
      all_hooks.reset()
      self.patcher.patch_hooks(None)

      for scheduled_opts in scheduled_keyframes:
        t_range = scheduled_opts[0]
        # don't bother encoding any conds outside of start_percent and end_percent bounds
        if "start_percent" in add_dict:
          if t_range[1] < add_dict["start_percent"]:
            continue
        if "end_percent" in add_dict:
          if t_range[0] > add_dict["end_percent"]:
            continue
        hooks_keyframes = scheduled_opts[1]
        for hook, keyframe in hooks_keyframes:
          hook.hook_keyframe._current_keyframe = keyframe
        # apply appropriate hooks with values that match new hook_keyframe
        self.patcher.patch_hooks(all_hooks)
        # perform encoding as normal
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        pooled_dict = {"pooled_output": pooled}
        # add clip_start_percent and clip_end_percent in pooled
        pooled_dict["clip_start_percent"] = t_range[0]
        pooled_dict["clip_end_percent"] = t_range[1]
        # add/update any keys with the provided add_dict
        pooled_dict.update(add_dict)
        # add hooks stored on clip
        self.add_hooks_to_dict(pooled_dict)
        all_cond_pooled.append([cond, pooled_dict])
        model_management.throw_exception_if_processing_interrupted()
      all_hooks.reset()
    return all_cond_pooled

  def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
    self.cond_stage_model.reset_clip_options()

    if self.layer_idx is not None:
      self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

    if return_pooled == "unprojected":
      self.cond_stage_model.set_clip_options({"projected_pooled": False})

    self.load_model()
    o = self.cond_stage_model.encode_token_weights(tokens)
    cond, pooled = o[:2]
    if return_dict:
      out = {"cond": cond, "pooled_output": pooled}
      if len(o) > 2:
        for k in o[2]:
          out[k] = o[2][k]
      self.add_hooks_to_dict(out)
      return out

    if return_pooled:
      return cond, pooled
    return cond

  def encode(self, text):
    tokens = self.tokenize(text)
    return self.encode_from_tokens(tokens)

  def load_sd(self, sd, full_model=False):
    if full_model:
      return self.cond_stage_model.load_state_dict(sd, strict=False)
    else:
      return self.cond_stage_model.load_sd(sd)

  def get_sd(self):
    sd_clip = self.cond_stage_model.state_dict()
    sd_tokenizer = self.tokenizer.state_dict()
    for k in sd_tokenizer:
      sd_clip[k] = sd_tokenizer[k]
    return sd_clip

  def load_model(self):
    model_management.load_model_gpu(self.patcher)
    return self.patcher

  def get_key_patches(self):
    return self.patcher.get_key_patches()


class CLIPType(Enum):
  STABLE_DIFFUSION = 1
  STABLE_CASCADE = 2
  SD3 = 3
  STABLE_AUDIO = 4
  HUNYUAN_DIT = 5
  FLUX = 6
  MOCHI = 7
  LTXV = 8
  HUNYUAN_VIDEO = 9
  PIXART = 10
  COSMOS = 11
  LUMINA2 = 12
  WAN = 13
  HIDREAM = 14
  CHROMA = 15
  ACE = 16
  OMNIGEN2 = 17
  QWEN_IMAGE = 18
  HUNYUAN_IMAGE = 19


def load_text_encoder_state_dicts(
  state_dicts=[],
  embedding_directory=None,
  clip_type=CLIPType.STABLE_DIFFUSION,
  model_options={},
):
  clip_data = state_dicts

  class EmptyClass:
    pass

  for i in range(len(clip_data)):
    if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
      clip_data[i] = utils.clip_text_transformers_convert(clip_data[i], "", "")
    else:
      if "text_projection" in clip_data[i]:
        clip_data[i]["text_projection.weight"] = clip_data[i][
          "text_projection"
        ].transpose(0, 1)  # old models saved with the CLIPSave node

  tokenizer_data = {}
  clip_target = EmptyClass()
  clip_target.params = {}
  if len(clip_data) == 1:
    te_model = detect_te_model(clip_data[0])
    if te_model == TEModel.T5_XXL:
      if clip_type == CLIPType.WAN:
        clip_target.clip = text_encoders.wan.te(**t5xxl_detect(clip_data))
        clip_target.tokenizer = text_encoders.wan.WanT5Tokenizer
        tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
      else:
        print("Unsupported CLIP type: ", clip_type)
    else:
      print("Unsupported TE model: ", te_model)
  else:
    print("Unsupported TE model: ", te_model)

  parameters = 0
  for c in clip_data:
    parameters += utils.calculate_parameters(c)
    tokenizer_data, model_options = text_encoders.long_clipl.model_options_long_clip(
      c, tokenizer_data, model_options
    )

  clip = CLIP(
    clip_target,
    embedding_directory=embedding_directory,
    parameters=parameters,
    tokenizer_data=tokenizer_data,
    model_options=model_options,
  )
  for c in clip_data:
    m, u = clip.load_sd(c)
    if len(m) > 0:
      logging.warning("clip missing: {}".format(m))

    if len(u) > 0:
      logging.debug("clip unexpected: {}".format(u))
  return clip


def load_clip(
  ckpt_paths,
  embedding_directory=None,
  clip_type=CLIPType.STABLE_DIFFUSION,
  model_options={},
):
  clip_data = []
  for p in ckpt_paths:
    clip_data.append(utils.load_torch_file(p, safe_load=True))
  return load_text_encoder_state_dicts(
    clip_data,
    embedding_directory=embedding_directory,
    clip_type=clip_type,
    model_options=model_options,
  )


def load_diffusion_model_state_dict(sd, model_options={}, metadata=None):
  """
  Loads a UNet diffusion model from a state dictionary, supporting both diffusers and regular formats.

  Args:
      sd (dict): State dictionary containing model weights and configuration
      model_options (dict, optional): Additional options for model loading. Supports:
          - dtype: Override model data type
          - custom_operations: Custom model operations
          - fp8_optimizations: Enable FP8 optimizations

  Returns:
      ModelPatcher: A wrapped model instance that handles device management and weight loading.
      Returns None if the model configuration cannot be detected.

  The function:
  1. Detects and handles different model formats (regular, diffusers, mmdit)
  2. Configures model dtype based on parameters and device capabilities
  3. Handles weight conversion and device placement
  4. Manages model optimization settings
  5. Loads weights and returns a device-managed model instance
  """
  dtype = model_options.get("dtype", None)

  # Allow loading unets from checkpoint files
  diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
  temp_sd = utils.state_dict_prefix_replace(
    sd, {diffusion_model_prefix: ""}, filter_keys=True
  )
  if len(temp_sd) > 0:
    sd = temp_sd

  parameters = utils.calculate_parameters(sd)
  weight_dtype = utils.weight_dtype(sd)

  load_device = model_management.get_torch_device()
  model_config = model_detection.model_config_from_unet(sd, "", metadata=metadata)

  if model_config is not None:
    new_sd = sd
  else:
    new_sd = model_detection.convert_diffusers_mmdit(sd, "")
    if new_sd is not None:  # diffusers mmdit
      model_config = model_detection.model_config_from_unet(new_sd, "")
      if model_config is None:
        return None
    else:  # diffusers unet
      model_config = model_detection.model_config_from_diffusers_unet(sd)
      if model_config is None:
        return None

      diffusers_keys = utils.unet_to_diffusers(model_config.unet_config)

      new_sd = {}
      for k in diffusers_keys:
        if k in sd:
          new_sd[diffusers_keys[k]] = sd.pop(k)
        else:
          logging.warning("{} {}".format(diffusers_keys[k], k))

  offload_device = model_management.unet_offload_device()
  unet_weight_dtype = list(model_config.supported_inference_dtypes)
  if model_config.scaled_fp8 is not None:
    weight_dtype = None

  if dtype is None:
    unet_dtype = model_management.unet_dtype(
      model_params=parameters,
      supported_dtypes=unet_weight_dtype,
      weight_dtype=weight_dtype,
    )
  else:
    unet_dtype = dtype

  if model_config.layer_quant_config is not None:
    manual_cast_dtype = model_management.unet_manual_cast(
      None, load_device, model_config.supported_inference_dtypes
    )
  else:
    manual_cast_dtype = model_management.unet_manual_cast(
      unet_dtype, load_device, model_config.supported_inference_dtypes
    )
  model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
  model_config.custom_operations = model_options.get(
    "custom_operations", model_config.custom_operations
  )
  if model_options.get("fp8_optimizations", False):
    model_config.optimizations["fp8"] = True

  model = model_config.get_model(new_sd, "")
  model = model.to(offload_device)
  model.load_model_weights(new_sd, "")
  left_over = sd.keys()
  if len(left_over) > 0:
    logging.info("left over keys in diffusion model: {}".format(left_over))
  return model_patcher.ModelPatcher(
    model, load_device=load_device, offload_device=offload_device
  )


def model_detection_error_hint(path, state_dict):
  filename = os.path.basename(path)
  if "lora" in filename.lower():
    return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded with a lora loader node.."
  return ""


def load_diffusion_model(unet_path, model_options={}):
  sd, metadata = utils.load_torch_file(unet_path, return_metadata=True)
  model = load_diffusion_model_state_dict(
    sd, model_options=model_options, metadata=metadata
  )
  if model is None:
    logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
    raise RuntimeError(
      "ERROR: Could not detect model type of: {}\n{}".format(
        unet_path, model_detection_error_hint(unet_path, sd)
      )
    )
  return model
