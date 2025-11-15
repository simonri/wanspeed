import torch
import math
import logging
import src.model_management as model_management
import src.ops as ops

def slice_attention(q, k, v):
  r1 = torch.zeros_like(k, device=q.device)
  scale = int(q.shape[-1]) ** (-0.5)

  mem_free_total = model_management.get_free_memory(q.device)

  tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
  modifier = 3 if q.element_size() == 2 else 2.5
  mem_required = tensor_size * modifier
  steps = 1

  if mem_required > mem_free_total:
    steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

  while True:
    try:
      slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
      for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        s1 = torch.bmm(q[:, i:end], k) * scale

        s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
        del s1

        r1[:, :, i:end] = torch.bmm(v, s2)
        del s2
      break
    except model_management.OOM_EXCEPTION as e:
      model_management.soft_empty_cache(True)
      steps *= 2
      if steps > 128:
        raise e
      logging.warning(
        "out of memory error, increasing steps and trying again {}".format(steps)
      )

  return r1


def normal_attention(q, k, v):
  # compute attention
  orig_shape = q.shape
  b = orig_shape[0]
  c = orig_shape[1]

  q = q.reshape(b, c, -1)
  q = q.permute(0, 2, 1)  # b,hw,c
  k = k.reshape(b, c, -1)  # b,c,hw
  v = v.reshape(b, c, -1)

  r1 = slice_attention(q, k, v)
  h_ = r1.reshape(orig_shape)
  del r1
  return h_


def pytorch_attention(q, k, v):
  # compute attention
  orig_shape = q.shape
  B = orig_shape[0]
  C = orig_shape[1]
  q, k, v = map(
    lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
    (q, k, v),
  )

  try:
    out = ops.scaled_dot_product_attention(
      q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(2, 3).reshape(orig_shape)
  except model_management.OOM_EXCEPTION:
    logging.warning("scaled_dot_product_attention OOMed: switched to slice attention")
    out = slice_attention(
      q.view(B, -1, C),
      k.view(B, -1, C).transpose(1, 2),
      v.view(B, -1, C).transpose(1, 2),
    ).reshape(orig_shape)
  return out


def vae_attention():
  if model_management.pytorch_attention_enabled_vae():
    logging.info("Using pytorch attention in VAE")
    return pytorch_attention
  else:
    logging.info("Using split attention in VAE")
    return normal_attention
