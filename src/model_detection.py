import torch
import logging
import json
import math

import src.supported_models as supported_models
import src.supported_models_base as supported_models_base


def count_blocks(state_dict_keys, prefix_string):
  count = 0
  while True:
    c = False
    for k in state_dict_keys:
      if k.startswith(prefix_string.format(count)):
        c = True
        break
    if not c:
      break
    count += 1
  return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
  context_dim = None
  use_linear_in_transformer = False

  transformer_prefix = prefix + "1.transformer_blocks."
  transformer_keys = sorted(
    list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))
  )
  if len(transformer_keys) > 0:
    last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + "{}")
    context_dim = state_dict["{}0.attn2.to_k.weight".format(transformer_prefix)].shape[
      1
    ]
    use_linear_in_transformer = (
      len(state_dict["{}1.proj_in.weight".format(prefix)].shape) == 2
    )
    time_stack = (
      "{}1.time_stack.0.attn1.to_q.weight".format(prefix) in state_dict
      or "{}1.time_mix_blocks.0.attn1.to_q.weight".format(prefix) in state_dict
    )
    time_stack_cross = (
      "{}1.time_stack.0.attn2.to_q.weight".format(prefix) in state_dict
      or "{}1.time_mix_blocks.0.attn2.to_q.weight".format(prefix) in state_dict
    )
    return (
      last_transformer_depth,
      context_dim,
      use_linear_in_transformer,
      time_stack,
      time_stack_cross,
    )
  return None


def model_config_from_unet_config(unet_config, state_dict=None):
  for model_config in supported_models.models:
    if model_config.matches(unet_config, state_dict):
      return model_config(unet_config)

  logging.error("no match {}".format(unet_config))
  return None


def detect_layer_quantization(metadata):
  quant_key = "_quantization_metadata"
  if metadata is not None and quant_key in metadata:
    quant_metadata = metadata.pop(quant_key)
    quant_metadata = json.loads(quant_metadata)
    if isinstance(quant_metadata, dict) and "layers" in quant_metadata:
      logging.info(
        f"Found quantization metadata (version {quant_metadata.get('format_version', 'unknown')})"
      )
      return quant_metadata["layers"]
    else:
      raise ValueError("Invalid quantization metadata format")
  return None


def detect_unet_config(state_dict, key_prefix, metadata=None):
  state_dict_keys = list(state_dict.keys())

  if (
    "{}joint_blocks.0.context_block.attn.qkv.weight".format(key_prefix)
    in state_dict_keys
  ):  # mmdit model
    unet_config = {}
    unet_config["in_channels"] = state_dict[
      "{}x_embedder.proj.weight".format(key_prefix)
    ].shape[1]
    patch_size = state_dict["{}x_embedder.proj.weight".format(key_prefix)].shape[2]
    unet_config["patch_size"] = patch_size
    final_layer = "{}final_layer.linear.weight".format(key_prefix)
    if final_layer in state_dict:
      unet_config["out_channels"] = state_dict[final_layer].shape[0] // (
        patch_size * patch_size
      )

    unet_config["depth"] = (
      state_dict["{}x_embedder.proj.weight".format(key_prefix)].shape[0] // 64
    )
    unet_config["input_size"] = None
    y_key = "{}y_embedder.mlp.0.weight".format(key_prefix)
    if y_key in state_dict_keys:
      unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

    context_key = "{}context_embedder.weight".format(key_prefix)
    if context_key in state_dict_keys:
      in_features = state_dict[context_key].shape[1]
      out_features = state_dict[context_key].shape[0]
      unet_config["context_embedder_config"] = {
        "target": "torch.nn.Linear",
        "params": {"in_features": in_features, "out_features": out_features},
      }
    num_patches_key = "{}pos_embed".format(key_prefix)
    if num_patches_key in state_dict_keys:
      num_patches = state_dict[num_patches_key].shape[1]
      unet_config["num_patches"] = num_patches
      unet_config["pos_embed_max_size"] = round(math.sqrt(num_patches))

    rms_qk = "{}joint_blocks.0.context_block.attn.ln_q.weight".format(key_prefix)
    if rms_qk in state_dict_keys:
      unet_config["qk_norm"] = "rms"

    unet_config["pos_embed_scaling_factor"] = None  # unused for inference
    context_processor = "{}context_processor.layers.0.attn.qkv.weight".format(
      key_prefix
    )
    if context_processor in state_dict_keys:
      unet_config["context_processor_layers"] = count_blocks(
        state_dict_keys, "{}context_processor.layers.".format(key_prefix) + "{}."
      )
    unet_config["x_block_self_attn_layers"] = []
    for key in state_dict_keys:
      if key.startswith("{}joint_blocks.".format(key_prefix)) and key.endswith(
        ".x_block.attn2.qkv.weight"
      ):
        layer = key[
          len("{}joint_blocks.".format(key_prefix)) : -len(".x_block.attn2.qkv.weight")
        ]
        unet_config["x_block_self_attn_layers"].append(int(layer))
    return unet_config

  if "{}clf.1.weight".format(key_prefix) in state_dict_keys:  # stable cascade
    unet_config = {}
    text_mapper_name = "{}clip_txt_mapper.weight".format(key_prefix)
    if text_mapper_name in state_dict_keys:
      unet_config["stable_cascade_stage"] = "c"
      w = state_dict[text_mapper_name]
      if w.shape[0] == 1536:  # stage c lite
        unet_config["c_cond"] = 1536
        unet_config["c_hidden"] = [1536, 1536]
        unet_config["nhead"] = [24, 24]
        unet_config["blocks"] = [[4, 12], [12, 4]]
      elif w.shape[0] == 2048:  # stage c full
        unet_config["c_cond"] = 2048
    elif "{}clip_mapper.weight".format(key_prefix) in state_dict_keys:
      unet_config["stable_cascade_stage"] = "b"
      w = state_dict["{}down_blocks.1.0.channelwise.0.weight".format(key_prefix)]
      if w.shape[-1] == 640:
        unet_config["c_hidden"] = [320, 640, 1280, 1280]
        unet_config["nhead"] = [-1, -1, 20, 20]
        unet_config["blocks"] = [[2, 6, 28, 6], [6, 28, 6, 2]]
        unet_config["block_repeat"] = [[1, 1, 1, 1], [3, 3, 2, 2]]
      elif w.shape[-1] == 576:  # stage b lite
        unet_config["c_hidden"] = [320, 576, 1152, 1152]
        unet_config["nhead"] = [-1, 9, 18, 18]
        unet_config["blocks"] = [[2, 4, 14, 4], [4, 14, 4, 2]]
        unet_config["block_repeat"] = [[1, 1, 1, 1], [2, 2, 2, 2]]
    return unet_config

  if (
    "{}transformer.rotary_pos_emb.inv_freq".format(key_prefix) in state_dict_keys
  ):  # stable audio dit
    unet_config = {}
    unet_config["audio_model"] = "dit1.0"
    return unet_config

  if (
    "{}double_layers.0.attn.w1q.weight".format(key_prefix) in state_dict_keys
  ):  # aura flow dit
    unet_config = {}
    unet_config["max_seq"] = state_dict[
      "{}positional_encoding".format(key_prefix)
    ].shape[1]
    unet_config["cond_seq_dim"] = state_dict[
      "{}cond_seq_linear.weight".format(key_prefix)
    ].shape[1]
    double_layers = count_blocks(
      state_dict_keys, "{}double_layers.".format(key_prefix) + "{}."
    )
    single_layers = count_blocks(
      state_dict_keys, "{}single_layers.".format(key_prefix) + "{}."
    )
    unet_config["n_double_layers"] = double_layers
    unet_config["n_layers"] = double_layers + single_layers
    return unet_config

  if "{}mlp_t5.0.weight".format(key_prefix) in state_dict_keys:  # Hunyuan DiT
    unet_config = {}
    unet_config["image_model"] = "hydit"
    unet_config["depth"] = count_blocks(
      state_dict_keys, "{}blocks.".format(key_prefix) + "{}."
    )
    unet_config["hidden_size"] = state_dict[
      "{}x_embedder.proj.weight".format(key_prefix)
    ].shape[0]
    if unet_config["hidden_size"] == 1408 and unet_config["depth"] == 40:  # DiT-g/2
      unet_config["mlp_ratio"] = 4.3637
    if state_dict["{}extra_embedder.0.weight".format(key_prefix)].shape[1] == 3968:
      unet_config["size_cond"] = True
      unet_config["use_style_cond"] = True
      unet_config["image_model"] = "hydit1"
    return unet_config

  if (
    "{}txt_in.individual_token_refiner.blocks.0.norm1.weight".format(key_prefix)
    in state_dict_keys
  ):  # Hunyuan Video
    dit_config = {}
    in_w = state_dict["{}img_in.proj.weight".format(key_prefix)]
    out_w = state_dict["{}final_layer.linear.weight".format(key_prefix)]
    dit_config["image_model"] = "hunyuan_video"
    dit_config["in_channels"] = in_w.shape[
      1
    ]  # SkyReels img2video has 32 input channels
    dit_config["patch_size"] = list(in_w.shape[2:])
    dit_config["out_channels"] = out_w.shape[0] // math.prod(dit_config["patch_size"])
    if any(s.startswith("{}vector_in.".format(key_prefix)) for s in state_dict_keys):
      dit_config["vec_in_dim"] = 768
    else:
      dit_config["vec_in_dim"] = None

    if len(dit_config["patch_size"]) == 2:
      dit_config["axes_dim"] = [64, 64]
    else:
      dit_config["axes_dim"] = [16, 56, 56]

    if any(s.startswith("{}time_r_in.".format(key_prefix)) for s in state_dict_keys):
      dit_config["meanflow"] = True
    else:
      dit_config["meanflow"] = False

    dit_config["context_in_dim"] = state_dict[
      "{}txt_in.input_embedder.weight".format(key_prefix)
    ].shape[1]
    dit_config["hidden_size"] = in_w.shape[0]
    dit_config["mlp_ratio"] = 4.0
    dit_config["num_heads"] = in_w.shape[0] // 128
    dit_config["depth"] = count_blocks(
      state_dict_keys, "{}double_blocks.".format(key_prefix) + "{}."
    )
    dit_config["depth_single_blocks"] = count_blocks(
      state_dict_keys, "{}single_blocks.".format(key_prefix) + "{}."
    )
    dit_config["theta"] = 256
    dit_config["qkv_bias"] = True
    if "{}byt5_in.fc1.weight".format(key_prefix) in state_dict:
      dit_config["byt5"] = True
    else:
      dit_config["byt5"] = False

    guidance_keys = list(
      filter(
        lambda a: a.startswith("{}guidance_in.".format(key_prefix)), state_dict_keys
      )
    )
    dit_config["guidance_embed"] = len(guidance_keys) > 0
    return dit_config

  if "{}double_blocks.0.img_attn.norm.key_norm.scale".format(
    key_prefix
  ) in state_dict_keys and (
    "{}img_in.weight".format(key_prefix) in state_dict_keys
    or f"{key_prefix}distilled_guidance_layer.norms.0.scale" in state_dict_keys
  ):  # Flux, Chroma or Chroma Radiance (has no img_in.weight)
    dit_config = {}
    dit_config["image_model"] = "flux"
    dit_config["in_channels"] = 16
    patch_size = 2
    dit_config["patch_size"] = patch_size
    in_key = "{}img_in.weight".format(key_prefix)
    if in_key in state_dict_keys:
      dit_config["in_channels"] = state_dict[in_key].shape[1] // (
        patch_size * patch_size
      )
    dit_config["out_channels"] = 16
    vec_in_key = "{}vector_in.in_layer.weight".format(key_prefix)
    if vec_in_key in state_dict_keys:
      dit_config["vec_in_dim"] = state_dict[vec_in_key].shape[1]
    dit_config["context_in_dim"] = 4096
    dit_config["hidden_size"] = 3072
    dit_config["mlp_ratio"] = 4.0
    dit_config["num_heads"] = 24
    dit_config["depth"] = count_blocks(
      state_dict_keys, "{}double_blocks.".format(key_prefix) + "{}."
    )
    dit_config["depth_single_blocks"] = count_blocks(
      state_dict_keys, "{}single_blocks.".format(key_prefix) + "{}."
    )
    dit_config["axes_dim"] = [16, 56, 56]
    dit_config["theta"] = 10000
    dit_config["qkv_bias"] = True
    if (
      "{}distilled_guidance_layer.0.norms.0.scale".format(key_prefix) in state_dict_keys
      or "{}distilled_guidance_layer.norms.0.scale".format(key_prefix)
      in state_dict_keys
    ):  # Chroma
      dit_config["image_model"] = "chroma"
      dit_config["in_channels"] = 64
      dit_config["out_channels"] = 64
      dit_config["in_dim"] = 64
      dit_config["out_dim"] = 3072
      dit_config["hidden_dim"] = 5120
      dit_config["n_layers"] = 5
      if f"{key_prefix}nerf_blocks.0.norm.scale" in state_dict_keys:  # Chroma Radiance
        dit_config["image_model"] = "chroma_radiance"
        dit_config["in_channels"] = 3
        dit_config["out_channels"] = 3
        dit_config["patch_size"] = 16
        dit_config["nerf_hidden_size"] = 64
        dit_config["nerf_mlp_ratio"] = 4
        dit_config["nerf_depth"] = 4
        dit_config["nerf_max_freqs"] = 8
        dit_config["nerf_tile_size"] = 512
        dit_config["nerf_final_head_type"] = (
          "conv"
          if f"{key_prefix}nerf_final_layer_conv.norm.scale" in state_dict_keys
          else "linear"
        )
        dit_config["nerf_embedder_dtype"] = torch.float32
    else:
      dit_config["guidance_embed"] = (
        "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
      )
    return dit_config

  if "{}t5_yproj.weight".format(key_prefix) in state_dict_keys:  # Genmo mochi preview
    dit_config = {}
    dit_config["image_model"] = "mochi_preview"
    dit_config["depth"] = 48
    dit_config["patch_size"] = 2
    dit_config["num_heads"] = 24
    dit_config["hidden_size_x"] = 3072
    dit_config["hidden_size_y"] = 1536
    dit_config["mlp_ratio_x"] = 4.0
    dit_config["mlp_ratio_y"] = 4.0
    dit_config["learn_sigma"] = False
    dit_config["in_channels"] = 12
    dit_config["qk_norm"] = True
    dit_config["qkv_bias"] = False
    dit_config["out_bias"] = True
    dit_config["attn_drop"] = 0.0
    dit_config["patch_embed_bias"] = True
    dit_config["posenc_preserve_area"] = True
    dit_config["timestep_mlp_bias"] = True
    dit_config["attend_to_padding"] = False
    dit_config["timestep_scale"] = 1000.0
    dit_config["use_t5"] = True
    dit_config["t5_feat_dim"] = 4096
    dit_config["t5_token_length"] = 256
    dit_config["rope_theta"] = 10000.0
    return dit_config

  if (
    "{}adaln_single.emb.timestep_embedder.linear_1.bias".format(key_prefix)
    in state_dict_keys
    and "{}pos_embed.proj.bias".format(key_prefix) in state_dict_keys
  ):
    # PixArt diffusers
    return None

  if (
    "{}adaln_single.emb.timestep_embedder.linear_1.bias".format(key_prefix)
    in state_dict_keys
  ):  # Lightricks ltxv
    dit_config = {}
    dit_config["image_model"] = "ltxv"
    dit_config["num_layers"] = count_blocks(
      state_dict_keys, "{}transformer_blocks.".format(key_prefix) + "{}."
    )
    shape = state_dict[
      "{}transformer_blocks.0.attn2.to_k.weight".format(key_prefix)
    ].shape
    dit_config["attention_head_dim"] = shape[0] // 32
    dit_config["cross_attention_dim"] = shape[1]
    if metadata is not None and "config" in metadata:
      dit_config.update(json.loads(metadata["config"]).get("transformer", {}))
    return dit_config

  if "{}genre_embedder.weight".format(key_prefix) in state_dict_keys:  # ACE-Step model
    dit_config = {}
    dit_config["audio_model"] = "ace"
    dit_config["attention_head_dim"] = 128
    dit_config["in_channels"] = 8
    dit_config["inner_dim"] = 2560
    dit_config["max_height"] = 16
    dit_config["max_position"] = 32768
    dit_config["max_width"] = 32768
    dit_config["mlp_ratio"] = 2.5
    dit_config["num_attention_heads"] = 20
    dit_config["num_layers"] = 24
    dit_config["out_channels"] = 8
    dit_config["patch_size"] = [16, 1]
    dit_config["rope_theta"] = 1000000.0
    dit_config["speaker_embedding_dim"] = 512
    dit_config["text_embedding_dim"] = 768

    dit_config["ssl_encoder_depths"] = [8, 8]
    dit_config["ssl_latent_dims"] = [1024, 768]
    dit_config["ssl_names"] = ["mert", "m-hubert"]
    dit_config["lyric_encoder_vocab_size"] = 6693
    dit_config["lyric_hidden_size"] = 1024
    return dit_config

  if "{}t_block.1.weight".format(key_prefix) in state_dict_keys:  # PixArt
    patch_size = 2
    dit_config = {}
    dit_config["num_heads"] = 16
    dit_config["patch_size"] = patch_size
    dit_config["hidden_size"] = 1152
    dit_config["in_channels"] = 4
    dit_config["depth"] = count_blocks(
      state_dict_keys, "{}blocks.".format(key_prefix) + "{}."
    )

    y_key = "{}y_embedder.y_embedding".format(key_prefix)
    if y_key in state_dict_keys:
      dit_config["model_max_length"] = state_dict[y_key].shape[0]

    pe_key = "{}pos_embed".format(key_prefix)
    if pe_key in state_dict_keys:
      dit_config["input_size"] = (
        int(math.sqrt(state_dict[pe_key].shape[1])) * patch_size
      )
      dit_config["pe_interpolation"] = dit_config["input_size"] // (512 // 8)  # guess

    ar_key = "{}ar_embedder.mlp.0.weight".format(key_prefix)
    if ar_key in state_dict_keys:
      dit_config["image_model"] = "pixart_alpha"
      dit_config["micro_condition"] = True
    else:
      dit_config["image_model"] = "pixart_sigma"
      dit_config["micro_condition"] = False
    return dit_config

  if (
    "{}blocks.block0.blocks.0.block.attn.to_q.0.weight".format(key_prefix)
    in state_dict_keys
  ):  # Cosmos
    dit_config = {}
    dit_config["image_model"] = "cosmos"
    dit_config["max_img_h"] = 240
    dit_config["max_img_w"] = 240
    dit_config["max_frames"] = 128
    concat_padding_mask = True
    dit_config["in_channels"] = (
      state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[1] // 4
    ) - int(concat_padding_mask)
    dit_config["out_channels"] = 16
    dit_config["patch_spatial"] = 2
    dit_config["patch_temporal"] = 1
    dit_config["model_channels"] = state_dict[
      "{}blocks.block0.blocks.0.block.attn.to_q.0.weight".format(key_prefix)
    ].shape[0]
    dit_config["block_config"] = "FA-CA-MLP"
    dit_config["concat_padding_mask"] = concat_padding_mask
    dit_config["pos_emb_cls"] = "rope3d"
    dit_config["pos_emb_learnable"] = False
    dit_config["pos_emb_interpolation"] = "crop"
    dit_config["block_x_format"] = "THWBD"
    dit_config["affline_emb_norm"] = True
    dit_config["use_adaln_lora"] = True
    dit_config["adaln_lora_dim"] = 256

    if dit_config["model_channels"] == 4096:
      # 7B
      dit_config["num_blocks"] = 28
      dit_config["num_heads"] = 32
      dit_config["extra_per_block_abs_pos_emb"] = True
      dit_config["rope_h_extrapolation_ratio"] = 1.0
      dit_config["rope_w_extrapolation_ratio"] = 1.0
      dit_config["rope_t_extrapolation_ratio"] = 2.0
      dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
    else:  # 5120
      # 14B
      dit_config["num_blocks"] = 36
      dit_config["num_heads"] = 40
      dit_config["extra_per_block_abs_pos_emb"] = True
      dit_config["rope_h_extrapolation_ratio"] = 2.0
      dit_config["rope_w_extrapolation_ratio"] = 2.0
      dit_config["rope_t_extrapolation_ratio"] = 2.0
      dit_config["extra_h_extrapolation_ratio"] = 2.0
      dit_config["extra_w_extrapolation_ratio"] = 2.0
      dit_config["extra_t_extrapolation_ratio"] = 2.0
      dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
    return dit_config

  if "{}cap_embedder.1.weight".format(key_prefix) in state_dict_keys:  # Lumina 2
    dit_config = {}
    dit_config["image_model"] = "lumina2"
    dit_config["patch_size"] = 2
    dit_config["in_channels"] = 16
    dit_config["dim"] = 2304
    dit_config["cap_feat_dim"] = state_dict[
      "{}cap_embedder.1.weight".format(key_prefix)
    ].shape[1]
    dit_config["n_layers"] = count_blocks(
      state_dict_keys, "{}layers.".format(key_prefix) + "{}."
    )
    dit_config["n_heads"] = 24
    dit_config["n_kv_heads"] = 8
    dit_config["qk_norm"] = True
    dit_config["axes_dims"] = [32, 32, 32]
    dit_config["axes_lens"] = [300, 512, 512]
    return dit_config

  if "{}head.modulation".format(key_prefix) in state_dict_keys:  # Wan 2.1
    dit_config = {}
    dit_config["image_model"] = "wan2.1"
    dim = state_dict["{}head.modulation".format(key_prefix)].shape[-1]
    out_dim = state_dict["{}head.head.weight".format(key_prefix)].shape[0] // 4
    dit_config["dim"] = dim
    dit_config["out_dim"] = out_dim
    dit_config["num_heads"] = dim // 128
    dit_config["ffn_dim"] = state_dict[
      "{}blocks.0.ffn.0.weight".format(key_prefix)
    ].shape[0]
    dit_config["num_layers"] = count_blocks(
      state_dict_keys, "{}blocks.".format(key_prefix) + "{}."
    )
    dit_config["patch_size"] = (1, 2, 2)
    dit_config["freq_dim"] = 256
    dit_config["window_size"] = (-1, -1)
    dit_config["qk_norm"] = True
    dit_config["cross_attn_norm"] = True
    dit_config["eps"] = 1e-6
    dit_config["in_dim"] = state_dict[
      "{}patch_embedding.weight".format(key_prefix)
    ].shape[1]
    if "{}vace_patch_embedding.weight".format(key_prefix) in state_dict_keys:
      dit_config["model_type"] = "vace"
      dit_config["vace_in_dim"] = state_dict[
        "{}vace_patch_embedding.weight".format(key_prefix)
      ].shape[1]
      dit_config["vace_layers"] = count_blocks(
        state_dict_keys, "{}vace_blocks.".format(key_prefix) + "{}."
      )
    elif "{}control_adapter.conv.weight".format(key_prefix) in state_dict_keys:
      if "{}img_emb.proj.0.bias".format(key_prefix) in state_dict_keys:
        dit_config["model_type"] = "camera"
      else:
        dit_config["model_type"] = "camera_2.2"
    elif (
      "{}casual_audio_encoder.encoder.final_linear.weight".format(key_prefix)
      in state_dict_keys
    ):
      dit_config["model_type"] = "s2v"
    elif (
      "{}audio_proj.audio_proj_glob_1.layer.bias".format(key_prefix) in state_dict_keys
    ):
      dit_config["model_type"] = "humo"
    elif (
      "{}face_adapter.fuser_blocks.0.k_norm.weight".format(key_prefix)
      in state_dict_keys
    ):
      dit_config["model_type"] = "animate"
    else:
      if "{}img_emb.proj.0.bias".format(key_prefix) in state_dict_keys:
        dit_config["model_type"] = "i2v"
      else:
        dit_config["model_type"] = "t2v"
    flf_weight = state_dict.get("{}img_emb.emb_pos".format(key_prefix))
    if flf_weight is not None:
      dit_config["flf_pos_embed_token_number"] = flf_weight.shape[1]

    ref_conv_weight = state_dict.get("{}ref_conv.weight".format(key_prefix))
    if ref_conv_weight is not None:
      dit_config["in_dim_ref_conv"] = ref_conv_weight.shape[1]

    return dit_config

  if "{}latent_in.weight".format(key_prefix) in state_dict_keys:  # Hunyuan 3D
    in_shape = state_dict["{}latent_in.weight".format(key_prefix)].shape
    dit_config = {}
    dit_config["image_model"] = "hunyuan3d2"
    dit_config["in_channels"] = in_shape[1]
    dit_config["context_in_dim"] = state_dict[
      "{}cond_in.weight".format(key_prefix)
    ].shape[1]
    dit_config["hidden_size"] = in_shape[0]
    dit_config["mlp_ratio"] = 4.0
    dit_config["num_heads"] = 16
    dit_config["depth"] = count_blocks(
      state_dict_keys, "{}double_blocks.".format(key_prefix) + "{}."
    )
    dit_config["depth_single_blocks"] = count_blocks(
      state_dict_keys, "{}single_blocks.".format(key_prefix) + "{}."
    )
    dit_config["qkv_bias"] = True
    dit_config["guidance_embed"] = (
      "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
    )
    return dit_config

  if f"{key_prefix}t_embedder.mlp.2.weight" in state_dict_keys:  # Hunyuan 3D 2.1
    dit_config = {}
    dit_config["image_model"] = "hunyuan3d2_1"
    dit_config["in_channels"] = state_dict[f"{key_prefix}x_embedder.weight"].shape[1]
    dit_config["context_dim"] = 1024
    dit_config["hidden_size"] = state_dict[f"{key_prefix}x_embedder.weight"].shape[0]
    dit_config["mlp_ratio"] = 4.0
    dit_config["num_heads"] = 16
    dit_config["depth"] = count_blocks(state_dict_keys, f"{key_prefix}blocks.{{}}")
    dit_config["qkv_bias"] = False
    dit_config["guidance_cond_proj_dim"] = (
      None  # f"{key_prefix}t_embedder.cond_proj.weight" in state_dict_keys
    )
    return dit_config

  if (
    "{}caption_projection.0.linear.weight".format(key_prefix) in state_dict_keys
  ):  # HiDream
    dit_config = {}
    dit_config["image_model"] = "hidream"
    dit_config["attention_head_dim"] = 128
    dit_config["axes_dims_rope"] = [64, 32, 32]
    dit_config["caption_channels"] = [4096, 4096]
    dit_config["max_resolution"] = [128, 128]
    dit_config["in_channels"] = 16
    dit_config["llama_layers"] = [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
      31,
    ]
    dit_config["num_attention_heads"] = 20
    dit_config["num_routed_experts"] = 4
    dit_config["num_activated_experts"] = 2
    dit_config["num_layers"] = 16
    dit_config["num_single_layers"] = 32
    dit_config["out_channels"] = 16
    dit_config["patch_size"] = 2
    dit_config["text_emb_dim"] = 2048
    return dit_config

  if (
    "{}blocks.0.mlp.layer1.weight".format(key_prefix) in state_dict_keys
  ):  # Cosmos predict2
    dit_config = {}
    dit_config["image_model"] = "cosmos_predict2"
    dit_config["max_img_h"] = 240
    dit_config["max_img_w"] = 240
    dit_config["max_frames"] = 128
    concat_padding_mask = True
    dit_config["in_channels"] = (
      state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[1] // 4
    ) - int(concat_padding_mask)
    dit_config["out_channels"] = 16
    dit_config["patch_spatial"] = 2
    dit_config["patch_temporal"] = 1
    dit_config["model_channels"] = state_dict[
      "{}x_embedder.proj.1.weight".format(key_prefix)
    ].shape[0]
    dit_config["concat_padding_mask"] = concat_padding_mask
    dit_config["crossattn_emb_channels"] = 1024
    dit_config["pos_emb_cls"] = "rope3d"
    dit_config["pos_emb_learnable"] = True
    dit_config["pos_emb_interpolation"] = "crop"
    dit_config["min_fps"] = 1
    dit_config["max_fps"] = 30

    dit_config["use_adaln_lora"] = True
    dit_config["adaln_lora_dim"] = 256
    if dit_config["model_channels"] == 2048:
      dit_config["num_blocks"] = 28
      dit_config["num_heads"] = 16
    elif dit_config["model_channels"] == 5120:
      dit_config["num_blocks"] = 36
      dit_config["num_heads"] = 40

    if dit_config["in_channels"] == 16:
      dit_config["extra_per_block_abs_pos_emb"] = False
      dit_config["rope_h_extrapolation_ratio"] = 4.0
      dit_config["rope_w_extrapolation_ratio"] = 4.0
      dit_config["rope_t_extrapolation_ratio"] = 1.0
    elif dit_config["in_channels"] == 17:  # img to video
      if dit_config["model_channels"] == 2048:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 3.0
        dit_config["rope_w_extrapolation_ratio"] = 3.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0
      elif dit_config["model_channels"] == 5120:
        dit_config["rope_h_extrapolation_ratio"] = 2.0
        dit_config["rope_w_extrapolation_ratio"] = 2.0
        dit_config["rope_t_extrapolation_ratio"] = 0.8333333333333334

    dit_config["extra_h_extrapolation_ratio"] = 1.0
    dit_config["extra_w_extrapolation_ratio"] = 1.0
    dit_config["extra_t_extrapolation_ratio"] = 1.0
    dit_config["rope_enable_fps_modulation"] = False

    return dit_config

  if (
    "{}time_caption_embed.timestep_embedder.linear_1.bias".format(key_prefix)
    in state_dict_keys
  ):  # Omnigen2
    dit_config = {}
    dit_config["image_model"] = "omnigen2"
    dit_config["axes_dim_rope"] = [40, 40, 40]
    dit_config["axes_lens"] = [1024, 1664, 1664]
    dit_config["ffn_dim_multiplier"] = None
    dit_config["hidden_size"] = 2520
    dit_config["in_channels"] = 16
    dit_config["multiple_of"] = 256
    dit_config["norm_eps"] = 1e-05
    dit_config["num_attention_heads"] = 21
    dit_config["num_kv_heads"] = 7
    dit_config["num_layers"] = 32
    dit_config["num_refiner_layers"] = 2
    dit_config["out_channels"] = None
    dit_config["patch_size"] = 2
    dit_config["text_feat_dim"] = 2048
    dit_config["timestep_scale"] = 1000.0
    return dit_config

  if "{}txt_norm.weight".format(key_prefix) in state_dict_keys:  # Qwen Image
    dit_config = {}
    dit_config["image_model"] = "qwen_image"
    dit_config["in_channels"] = state_dict["{}img_in.weight".format(key_prefix)].shape[
      1
    ]
    dit_config["num_layers"] = count_blocks(
      state_dict_keys, "{}transformer_blocks.".format(key_prefix) + "{}."
    )
    return dit_config

  if "{}input_blocks.0.0.weight".format(key_prefix) not in state_dict_keys:
    return None

  unet_config = {
    "use_checkpoint": False,
    "image_size": 32,
    "use_spatial_transformer": True,
    "legacy": False,
  }

  y_input = "{}label_emb.0.0.weight".format(key_prefix)
  if y_input in state_dict_keys:
    unet_config["num_classes"] = "sequential"
    unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
  else:
    unet_config["adm_in_channels"] = None

  model_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[0]
  in_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[1]

  out_key = "{}out.2.weight".format(key_prefix)
  if out_key in state_dict:
    out_channels = state_dict[out_key].shape[0]
  else:
    out_channels = 4

  num_res_blocks = []
  channel_mult = []
  transformer_depth = []
  transformer_depth_output = []
  context_dim = None
  use_linear_in_transformer = False

  video_model = False
  video_model_cross = False

  current_res = 1
  count = 0

  last_res_blocks = 0
  last_channel_mult = 0

  input_block_count = count_blocks(
    state_dict_keys, "{}input_blocks".format(key_prefix) + ".{}."
  )
  for count in range(input_block_count):
    prefix = "{}input_blocks.{}.".format(key_prefix, count)
    prefix_output = "{}output_blocks.{}.".format(
      key_prefix, input_block_count - count - 1
    )

    block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
    if len(block_keys) == 0:
      break

    block_keys_output = sorted(
      list(filter(lambda a: a.startswith(prefix_output), state_dict_keys))
    )

    if "{}0.op.weight".format(prefix) in block_keys:  # new layer
      num_res_blocks.append(last_res_blocks)
      channel_mult.append(last_channel_mult)

      current_res *= 2
      last_res_blocks = 0
      last_channel_mult = 0
      out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
      if out is not None:
        transformer_depth_output.append(out[0])
      else:
        transformer_depth_output.append(0)
    else:
      res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
      if res_block_prefix in block_keys:
        last_res_blocks += 1
        last_channel_mult = (
          state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0]
          // model_channels
        )

        out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
        if out is not None:
          transformer_depth.append(out[0])
          if context_dim is None:
            context_dim = out[1]
            use_linear_in_transformer = out[2]
            video_model = out[3]
            video_model_cross = out[4]
        else:
          transformer_depth.append(0)

      res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
      if res_block_prefix in block_keys_output:
        out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
        if out is not None:
          transformer_depth_output.append(out[0])
        else:
          transformer_depth_output.append(0)

  num_res_blocks.append(last_res_blocks)
  channel_mult.append(last_channel_mult)
  if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
    transformer_depth_middle = count_blocks(
      state_dict_keys, "{}middle_block.1.transformer_blocks.".format(key_prefix) + "{}"
    )
  elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
    transformer_depth_middle = -1
  else:
    transformer_depth_middle = -2

  unet_config["in_channels"] = in_channels
  unet_config["out_channels"] = out_channels
  unet_config["model_channels"] = model_channels
  unet_config["num_res_blocks"] = num_res_blocks
  unet_config["transformer_depth"] = transformer_depth
  unet_config["transformer_depth_output"] = transformer_depth_output
  unet_config["channel_mult"] = channel_mult
  unet_config["transformer_depth_middle"] = transformer_depth_middle
  unet_config["use_linear_in_transformer"] = use_linear_in_transformer
  unet_config["context_dim"] = context_dim

  if video_model:
    unet_config["extra_ff_mix_layer"] = True
    unet_config["use_spatial_context"] = True
    unet_config["merge_strategy"] = "learned_with_images"
    unet_config["merge_factor"] = 0.0
    unet_config["video_kernel_size"] = [3, 1, 1]
    unet_config["use_temporal_resblock"] = True
    unet_config["use_temporal_attention"] = True
    unet_config["disable_temporal_crossattention"] = not video_model_cross
  else:
    unet_config["use_temporal_resblock"] = False
    unet_config["use_temporal_attention"] = False

  return unet_config


def model_config_from_unet(
  state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None
):
  unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
  if unet_config is None:
    return None
  model_config = model_config_from_unet_config(unet_config, state_dict)
  if model_config is None and use_base_if_no_match:
    model_config = supported_models_base.BASE(unet_config)

  scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
  if scaled_fp8_key in state_dict:
    scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
    model_config.scaled_fp8 = scaled_fp8_weight.dtype
    if model_config.scaled_fp8 == torch.float32:
      model_config.scaled_fp8 = torch.float8_e4m3fn
    if scaled_fp8_weight.nelement() == 2:
      model_config.optimizations["fp8"] = False
    else:
      model_config.optimizations["fp8"] = True

  # Detect per-layer quantization (mixed precision)
  layer_quant_config = detect_layer_quantization(metadata)
  if layer_quant_config:
    model_config.layer_quant_config = layer_quant_config
    logging.info(
      f"Detected mixed precision quantization: {len(layer_quant_config)} layers quantized"
    )

  return model_config
