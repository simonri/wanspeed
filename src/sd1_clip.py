import torch
import logging
import os
import json
import numbers
import src.model_management as model_management
import src.clip_model as clip_model
import src.ops as ops

def gen_empty_tokens(special_tokens, length):
  start_token = special_tokens.get("start", None)
  end_token = special_tokens.get("end", None)
  pad_token = special_tokens.get("pad")
  output = []
  if start_token is not None:
    output.append(start_token)
  if end_token is not None:
    output.append(end_token)
  output += [pad_token] * (length - len(output))
  return output


class ClipTokenWeightEncoder:
  def encode_token_weights(self, token_weight_pairs):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
      tokens = list(map(lambda a: a[0], x))
      max_token_len = max(len(tokens), max_token_len)
      has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
      to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
      if hasattr(self, "gen_empty_tokens"):
        to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
      else:
        to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

    o = self.encode(to_encode)
    out, pooled = o[:2]

    if pooled is not None:
      first_pooled = pooled[0:1].to(model_management.intermediate_device())
    else:
      first_pooled = pooled

    output = []
    for k in range(0, sections):
      z = out[k : k + 1]
      if has_weights:
        z_empty = out[-1]
        for i in range(len(z)):
          for j in range(len(z[i])):
            weight = token_weight_pairs[k][j][1]
            if weight != 1.0:
              z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
      output.append(z)

    if len(output) == 0:
      r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
    else:
      r = (
        torch.cat(output, dim=-2).to(model_management.intermediate_device()),
        first_pooled,
      )

    if len(o) > 2:
      extra = {}
      for k in o[2]:
        v = o[2][k]
        if k == "attention_mask":
          v = (
            v[:sections]
            .flatten()
            .unsqueeze(dim=0)
            .to(model_management.intermediate_device())
          )
        extra[k] = v

      r = r + (extra,)
    return r


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
  LAYERS = ["last", "pooled", "hidden", "all"]

  def __init__(
    self,
    device="cpu",
    max_length=77,
    freeze=True,
    layer="last",
    layer_idx=None,
    textmodel_json_config=None,
    dtype=None,
    model_class=clip_model.CLIPTextModel,
    special_tokens={"start": 49406, "end": 49407, "pad": 49407},
    layer_norm_hidden_state=True,
    enable_attention_masks=False,
    zero_out_masked=False,
    return_projected_pooled=True,
    return_attention_masks=False,
    model_options={},
  ):  # clip-vit-base-patch32
    super().__init__()
    assert layer in self.LAYERS

    if textmodel_json_config is None:
      textmodel_json_config = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sd1_clip_config.json"
      )
      if "model_name" not in model_options:
        model_options = {**model_options, "model_name": "clip_l"}

    if isinstance(textmodel_json_config, dict):
      config = textmodel_json_config
    else:
      with open(textmodel_json_config) as f:
        config = json.load(f)

    te_model_options = model_options.get(
      "{}_model_config".format(model_options.get("model_name", "")), {}
    )
    for k, v in te_model_options.items():
      config[k] = v

    operations = model_options.get("custom_operations", None)
    scaled_fp8 = None

    if operations is None:
      scaled_fp8 = model_options.get("scaled_fp8", None)
      if scaled_fp8 is not None:
        operations = ops.scaled_fp8_ops(
          fp8_matrix_mult=False, override_dtype=scaled_fp8
        )
      else:
        operations = ops.manual_cast

    self.operations = operations
    self.transformer = model_class(config, dtype, device, self.operations)
    if scaled_fp8 is not None:
      self.transformer.scaled_fp8 = torch.nn.Parameter(
        torch.tensor([], dtype=scaled_fp8)
      )

    self.num_layers = self.transformer.num_layers

    self.max_length = max_length
    if freeze:
      self.freeze()
    self.layer = layer
    self.layer_idx = None
    self.special_tokens = special_tokens

    self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
    self.enable_attention_masks = enable_attention_masks
    self.zero_out_masked = zero_out_masked

    self.layer_norm_hidden_state = layer_norm_hidden_state
    self.return_projected_pooled = return_projected_pooled
    self.return_attention_masks = return_attention_masks

    if layer == "hidden":
      assert layer_idx is not None
      assert abs(layer_idx) < self.num_layers
      self.set_clip_options({"layer": layer_idx})
    self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

  def freeze(self):
    self.transformer = self.transformer.eval()
    # self.train = disabled_train
    for param in self.parameters():
      param.requires_grad = False

  def set_clip_options(self, options):
    layer_idx = options.get("layer", self.layer_idx)
    self.return_projected_pooled = options.get(
      "projected_pooled", self.return_projected_pooled
    )
    if self.layer == "all":
      pass
    elif layer_idx is None or abs(layer_idx) > self.num_layers:
      self.layer = "last"
    else:
      self.layer = "hidden"
      self.layer_idx = layer_idx

  def reset_clip_options(self):
    self.layer = self.options_default[0]
    self.layer_idx = self.options_default[1]
    self.return_projected_pooled = self.options_default[2]

  def process_tokens(self, tokens, device):
    end_token = self.special_tokens.get("end", None)
    if end_token is None:
      cmp_token = self.special_tokens.get("pad", -1)
    else:
      cmp_token = end_token

    embeds_out = []
    attention_masks = []
    num_tokens = []

    for x in tokens:
      attention_mask = []
      tokens_temp = []
      other_embeds = []
      eos = False
      index = 0
      for y in x:
        if isinstance(y, numbers.Integral):
          if eos:
            attention_mask.append(0)
          else:
            attention_mask.append(1)
          token = int(y)
          tokens_temp += [token]
          if not eos and token == cmp_token:
            if end_token is None:
              attention_mask[-1] = 0
            eos = True
        else:
          other_embeds.append((index, y))
        index += 1

      tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
      tokens_embed = self.transformer.get_input_embeddings()(
        tokens_embed, out_dtype=torch.float32
      )
      index = 0
      pad_extra = 0
      embeds_info = []
      for o in other_embeds:
        emb = o[1]
        if torch.is_tensor(emb):
          emb = {"type": "embedding", "data": emb}

        extra = None
        emb_type = emb.get("type", None)
        if emb_type == "embedding":
          emb = emb.get("data", None)
        else:
          if hasattr(self.transformer, "preprocess_embed"):
            emb, extra = self.transformer.preprocess_embed(emb, device=device)
          else:
            emb = None

        if emb is None:
          index += -1
          continue

        ind = index + o[0]
        emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
        emb_shape = emb.shape[1]
        if emb.shape[-1] == tokens_embed.shape[-1]:
          tokens_embed = torch.cat(
            [tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1
          )
          attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
          index += emb_shape - 1
          embeds_info.append(
            {"type": emb_type, "index": ind, "size": emb_shape, "extra": extra}
          )
        else:
          index += -1
          pad_extra += emb_shape
          logging.warning(
            "WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(
              emb.shape[-1], tokens_embed.shape[-1]
            )
          )

      if pad_extra > 0:
        padd_embed = self.transformer.get_input_embeddings()(
          torch.tensor(
            [[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long
          ),
          out_dtype=torch.float32,
        )
        tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
        attention_mask = attention_mask + [0] * pad_extra

      embeds_out.append(tokens_embed)
      attention_masks.append(attention_mask)
      num_tokens.append(sum(attention_mask))

    return (
      torch.cat(embeds_out),
      torch.tensor(attention_masks, device=device, dtype=torch.long),
      num_tokens,
      embeds_info,
    )

  def forward(self, tokens):
    device = self.transformer.get_input_embeddings().weight.device
    embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(
      tokens, device
    )

    attention_mask_model = None
    if self.enable_attention_masks:
      attention_mask_model = attention_mask

    if self.layer == "all":
      intermediate_output = "all"
    else:
      intermediate_output = self.layer_idx

    outputs = self.transformer(
      None,
      attention_mask_model,
      embeds=embeds,
      num_tokens=num_tokens,
      intermediate_output=intermediate_output,
      final_layer_norm_intermediate=self.layer_norm_hidden_state,
      dtype=torch.float32,
      embeds_info=embeds_info,
    )

    if self.layer == "last":
      z = outputs[0].float()
    else:
      z = outputs[1].float()

    if self.zero_out_masked:
      z *= attention_mask.unsqueeze(-1).float()

    pooled_output = None
    if len(outputs) >= 3:
      if (
        not self.return_projected_pooled
        and len(outputs) >= 4
        and outputs[3] is not None
      ):
        pooled_output = outputs[3].float()
      elif outputs[2] is not None:
        pooled_output = outputs[2].float()

    extra = {}
    if self.return_attention_masks:
      extra["attention_mask"] = attention_mask

    if len(extra) > 0:
      return z, pooled_output, extra

    return z, pooled_output

  def encode(self, tokens):
    return self(tokens)

  def load_sd(self, sd):
    return self.transformer.load_state_dict(sd, strict=False)


class SD1CheckpointClipModel(SDClipModel):
  def __init__(self, device="cpu", dtype=None, model_options={}):
    super().__init__(
      device=device,
      return_projected_pooled=False,
      dtype=dtype,
      model_options=model_options,
    )


class SD1ClipModel(torch.nn.Module):
  def __init__(
    self,
    device="cpu",
    dtype=None,
    model_options={},
    clip_name="l",
    clip_model=SD1CheckpointClipModel,
    name=None,
    **kwargs,
  ):
    super().__init__()

    if name is not None:
      self.clip_name = name
      self.clip = "{}".format(self.clip_name)
    else:
      self.clip_name = clip_name
      self.clip = "clip_{}".format(self.clip_name)

    clip_model = model_options.get("{}_class".format(self.clip), clip_model)
    model_options = {**model_options, "model_name": self.clip}
    setattr(
      self,
      self.clip,
      clip_model(device=device, dtype=dtype, model_options=model_options, **kwargs),
    )

    self.dtypes = set()
    if dtype is not None:
      self.dtypes.add(dtype)

  def set_clip_options(self, options):
    getattr(self, self.clip).set_clip_options(options)

  def reset_clip_options(self):
    getattr(self, self.clip).reset_clip_options()

  def encode_token_weights(self, token_weight_pairs):
    token_weight_pairs = token_weight_pairs[self.clip_name]
    out = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
    return out

  def load_sd(self, sd):
    return getattr(self, self.clip).load_sd(sd)
