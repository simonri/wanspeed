import src.sd1_clip as sd1_clip
import os
import src.text_encoders as text_encoders


class UMT5XXlModel(sd1_clip.SDClipModel):
  def __init__(
    self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}
  ):
    textmodel_json_config = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "umt5_config_xxl.json"
    )
    super().__init__(
      device=device,
      layer=layer,
      layer_idx=layer_idx,
      textmodel_json_config=textmodel_json_config,
      dtype=dtype,
      special_tokens={"end": 1, "pad": 0},
      model_class=text_encoders.t5.T5,
      enable_attention_masks=True,
      zero_out_masked=True,
      model_options=model_options,
    )


class WanT5Model(sd1_clip.SD1ClipModel):
  def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
    super().__init__(
      device=device,
      dtype=dtype,
      model_options=model_options,
      name="umt5xxl",
      clip_model=UMT5XXlModel,
      **kwargs,
    )


def te(dtype_t5=None, t5xxl_scaled_fp8=None):
  class WanTEModel(WanT5Model):
    def __init__(self, device="cpu", dtype=None, model_options={}):
      if t5xxl_scaled_fp8 is not None and "scaled_fp8" not in model_options:
        model_options = model_options.copy()
        model_options["scaled_fp8"] = t5xxl_scaled_fp8
      if dtype_t5 is not None:
        dtype = dtype_t5
      super().__init__(device=device, dtype=dtype, model_options=model_options)

  return WanTEModel
