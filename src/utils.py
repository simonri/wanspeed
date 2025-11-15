import math
import torch
import numpy as np
from PIL import Image
import safetensors


def repeat_to_batch_size(tensor, batch_size, dim=0):
  if tensor.shape[dim] > batch_size:
    return tensor.narrow(dim, 0, batch_size)
  elif tensor.shape[dim] < batch_size:
    return tensor.repeat(
      dim * [1]
      + [math.ceil(batch_size / tensor.shape[dim])]
      + [1] * (len(tensor.shape) - 1 - dim)
    ).narrow(dim, 0, batch_size)
  return tensor


def bislerp(samples, width, height):
  def slerp(b1, b2, r):
    """slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC"""

    c = b1.shape[-1]

    # norms
    b1_norms = torch.norm(b1, dim=-1, keepdim=True)
    b2_norms = torch.norm(b2, dim=-1, keepdim=True)

    # normalize
    b1_normalized = b1 / b1_norms
    b2_normalized = b2 / b2_norms

    # zero when norms are zero
    b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
    b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

    # slerp
    dot = (b1_normalized * b2_normalized).sum(1)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    # technically not mathematically correct, but more pleasing?
    res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(
      1
    ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(
      1
    ) * b2_normalized
    res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

    # edge cases for same or polar opposites
    res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
    res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
    return res


def lanczos(samples, width, height):
  images = [
    Image.fromarray(
      np.clip(255.0 * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)
    )
    for image in samples
  ]
  images = [
    image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images
  ]
  images = [
    torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0)
    for image in images
  ]
  result = torch.stack(images)
  return result.to(samples.device, samples.dtype)


def common_upscale(samples, width, height, upscale_method, crop):
  orig_shape = tuple(samples.shape)
  if len(orig_shape) > 4:
    samples = samples.reshape(
      samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1]
    )
    samples = samples.movedim(2, 1)
    samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
  if crop == "center":
    old_width = samples.shape[-1]
    old_height = samples.shape[-2]
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = 0
    y = 0
    if old_aspect > new_aspect:
      x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
      y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
  else:
    s = samples

  if upscale_method == "bislerp":
    out = bislerp(s, width, height)
  elif upscale_method == "lanczos":
    out = lanczos(s, width, height)
  else:
    out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

  if len(orig_shape) == 4:
    return out

  out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
  return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
  if device is None:
    device = torch.device("cpu")
  metadata = None
  if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
    try:
      with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
        sd = {}
        for k in f.keys():
          tensor = f.get_tensor(k)
          sd[k] = tensor
        if return_metadata:
          metadata = f.metadata()
    except Exception as e:
      if len(e.args) > 0:
        message = e.args[0]
        if "HeaderTooLarge" in message:
          raise ValueError(
            "{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(
              message, ckpt
            )
          )
        if "MetadataIncompleteBuffer" in message:
          raise ValueError(
            "{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(
              message, ckpt
            )
          )
      raise e
  else:
    torch_args = {}
    pl_sd = torch.load(ckpt, map_location=device, weights_only=True, **torch_args)
    if "state_dict" in pl_sd:
      sd = pl_sd["state_dict"]
    else:
      if len(pl_sd) == 1:
        key = list(pl_sd.keys())[0]
        sd = pl_sd[key]
        if not isinstance(sd, dict):
          sd = pl_sd
      else:
        sd = pl_sd
  return (sd, metadata) if return_metadata else sd


def calculate_parameters(sd, prefix=""):
  params = 0
  for k in sd.keys():
    if k.startswith(prefix):
      w = sd[k]
      params += w.nelement()
  return params


def weight_dtype(sd, prefix=""):
  dtypes = {}
  for k in sd.keys():
    if k.startswith(prefix):
      w = sd[k]
      dtypes[w.dtype] = dtypes.get(w.dtype, 0) + w.numel()

  if len(dtypes) == 0:
    return None

  return max(dtypes, key=dtypes.get)


def copy_to_param(obj, attr, value):
  # inplace update tensor instead of replacing it
  attrs = attr.split(".")
  for name in attrs[:-1]:
    obj = getattr(obj, name)
  prev = getattr(obj, attrs[-1])
  prev.data.copy_(value)


def set_attr(obj, attr, value):
  attrs = attr.split(".")
  for name in attrs[:-1]:
    obj = getattr(obj, name)
  prev = getattr(obj, attrs[-1])
  setattr(obj, attrs[-1], value)
  return prev


def set_attr_param(obj, attr, value):
  return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
  if filter_keys:
    out = {}
  else:
    out = state_dict
  for rp in replace_prefix:
    replace = list(
      map(
        lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
        filter(lambda a: a.startswith(rp), state_dict.keys()),
      )
    )
    for x in replace:
      w = state_dict.pop(x[0])
      out[x[1]] = w
  return out


def swap_scale_shift(weight):
  shift, scale = weight.chunk(2, dim=0)
  new_weight = torch.cat([scale, shift], dim=0)
  return new_weight


MMDIT_MAP_BASIC = {
  ("context_embedder.bias", "context_embedder.bias"),
  ("context_embedder.weight", "context_embedder.weight"),
  ("t_embedder.mlp.0.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
  ("t_embedder.mlp.0.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
  ("t_embedder.mlp.2.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
  ("t_embedder.mlp.2.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
  ("x_embedder.proj.bias", "pos_embed.proj.bias"),
  ("x_embedder.proj.weight", "pos_embed.proj.weight"),
  ("y_embedder.mlp.0.bias", "time_text_embed.text_embedder.linear_1.bias"),
  ("y_embedder.mlp.0.weight", "time_text_embed.text_embedder.linear_1.weight"),
  ("y_embedder.mlp.2.bias", "time_text_embed.text_embedder.linear_2.bias"),
  ("y_embedder.mlp.2.weight", "time_text_embed.text_embedder.linear_2.weight"),
  ("pos_embed", "pos_embed.pos_embed"),
  ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", swap_scale_shift),
  ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
  ("final_layer.linear.bias", "proj_out.bias"),
  ("final_layer.linear.weight", "proj_out.weight"),
}


MMDIT_MAP_BLOCK = {
  ("context_block.adaLN_modulation.1.bias", "norm1_context.linear.bias"),
  ("context_block.adaLN_modulation.1.weight", "norm1_context.linear.weight"),
  ("context_block.attn.proj.bias", "attn.to_add_out.bias"),
  ("context_block.attn.proj.weight", "attn.to_add_out.weight"),
  ("context_block.mlp.fc1.bias", "ff_context.net.0.proj.bias"),
  ("context_block.mlp.fc1.weight", "ff_context.net.0.proj.weight"),
  ("context_block.mlp.fc2.bias", "ff_context.net.2.bias"),
  ("context_block.mlp.fc2.weight", "ff_context.net.2.weight"),
  ("context_block.attn.ln_q.weight", "attn.norm_added_q.weight"),
  ("context_block.attn.ln_k.weight", "attn.norm_added_k.weight"),
  ("x_block.adaLN_modulation.1.bias", "norm1.linear.bias"),
  ("x_block.adaLN_modulation.1.weight", "norm1.linear.weight"),
  ("x_block.attn.proj.bias", "attn.to_out.0.bias"),
  ("x_block.attn.proj.weight", "attn.to_out.0.weight"),
  ("x_block.attn.ln_q.weight", "attn.norm_q.weight"),
  ("x_block.attn.ln_k.weight", "attn.norm_k.weight"),
  ("x_block.attn2.proj.bias", "attn2.to_out.0.bias"),
  ("x_block.attn2.proj.weight", "attn2.to_out.0.weight"),
  ("x_block.attn2.ln_q.weight", "attn2.norm_q.weight"),
  ("x_block.attn2.ln_k.weight", "attn2.norm_k.weight"),
  ("x_block.mlp.fc1.bias", "ff.net.0.proj.bias"),
  ("x_block.mlp.fc1.weight", "ff.net.0.proj.weight"),
  ("x_block.mlp.fc2.bias", "ff.net.2.bias"),
  ("x_block.mlp.fc2.weight", "ff.net.2.weight"),
}


def mmdit_to_diffusers(mmdit_config, output_prefix=""):
  key_map = {}

  depth = mmdit_config.get("depth", 0)
  num_blocks = mmdit_config.get("num_blocks", depth)
  for i in range(num_blocks):
    block_from = "transformer_blocks.{}".format(i)
    block_to = "{}joint_blocks.{}".format(output_prefix, i)

    offset = depth * 64

    for end in ("weight", "bias"):
      k = "{}.attn.".format(block_from)
      qkv = "{}.x_block.attn.qkv.{}".format(block_to, end)
      key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
      key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
      key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

      qkv = "{}.context_block.attn.qkv.{}".format(block_to, end)
      key_map["{}add_q_proj.{}".format(k, end)] = (qkv, (0, 0, offset))
      key_map["{}add_k_proj.{}".format(k, end)] = (qkv, (0, offset, offset))
      key_map["{}add_v_proj.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

      k = "{}.attn2.".format(block_from)
      qkv = "{}.x_block.attn2.qkv.{}".format(block_to, end)
      key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
      key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
      key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

    for k in MMDIT_MAP_BLOCK:
      key_map["{}.{}".format(block_from, k[1])] = "{}.{}".format(block_to, k[0])

  map_basic = MMDIT_MAP_BASIC.copy()
  map_basic.add(
    (
      "joint_blocks.{}.context_block.adaLN_modulation.1.bias".format(depth - 1),
      "transformer_blocks.{}.norm1_context.linear.bias".format(depth - 1),
      swap_scale_shift,
    )
  )
  map_basic.add(
    (
      "joint_blocks.{}.context_block.adaLN_modulation.1.weight".format(depth - 1),
      "transformer_blocks.{}.norm1_context.linear.weight".format(depth - 1),
      swap_scale_shift,
    )
  )

  for k in map_basic:
    if len(k) > 2:
      key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
    else:
      key_map[k[1]] = "{}{}".format(output_prefix, k[0])

  return key_map
