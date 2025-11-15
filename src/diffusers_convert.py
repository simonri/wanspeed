import logging

vae_conversion_map = [
  # (stable-diffusion, HF Diffusers)
  ("nin_shortcut", "conv_shortcut"),
  ("norm_out", "conv_norm_out"),
  ("mid.attn_1.", "mid_block.attentions.0."),
]

vae_conversion_map_attn = [
  # (stable-diffusion, HF Diffusers)
  ("norm.", "group_norm."),
  ("q.", "query."),
  ("k.", "key."),
  ("v.", "value."),
  ("q.", "to_q."),
  ("k.", "to_k."),
  ("v.", "to_v."),
  ("proj_out.", "to_out.0."),
  ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w, conv3d=False):
  # convert HF linear weights to SD conv2d weights
  if conv3d:
    return w.reshape(*w.shape, 1, 1, 1)
  else:
    return w.reshape(*w.shape, 1, 1)


def convert_vae_state_dict(vae_state_dict):
  mapping = {k: k for k in vae_state_dict.keys()}
  conv3d = False
  for k, v in mapping.items():
    for sd_part, hf_part in vae_conversion_map:
      v = v.replace(hf_part, sd_part)
    if v.endswith(".conv.weight"):
      if not conv3d and vae_state_dict[k].ndim == 5:
        conv3d = True
    mapping[k] = v
  for k, v in mapping.items():
    if "attentions" in k:
      for sd_part, hf_part in vae_conversion_map_attn:
        v = v.replace(hf_part, sd_part)
      mapping[k] = v
  new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
  weights_to_convert = ["q", "k", "v", "proj_out"]
  for k, v in new_state_dict.items():
    for weight_name in weights_to_convert:
      if f"mid.attn_1.{weight_name}.weight" in k:
        logging.debug(f"Reshaping {k} for SD format")
        new_state_dict[k] = reshape_weight_for_sd(v, conv3d=conv3d)
  return new_state_dict
