import torch
import src.latent_formats as latent_formats
import src.wan.model_base as model_base
import src.supported_models_base as supported_models_base
import src.text_encoders as text_encoders

class WAN21_T2V(supported_models_base.BASE):
  unet_config = {
    "image_model": "wan2.1",
    "model_type": "t2v",
  }

  sampling_settings = {
    "shift": 8.0,
  }

  unet_extra_config = {}
  latent_format = latent_formats.Wan21

  memory_usage_factor = 0.9

  supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

  vae_key_prefix = ["vae."]
  text_encoder_key_prefix = ["text_encoders."]

  def __init__(self, unet_config):
    super().__init__(unet_config)
    self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2222

  def get_model(self, state_dict, prefix="", device=None):
    out = model_base.WAN21(self, device=device)
    return out

  def clip_target(self, state_dict={}):
    pref = self.text_encoder_key_prefix[0]
    t5_detect = text_encoders.sd3_clip.t5_xxl_detect(
      state_dict, "{}umt5xxl.transformer.".format(pref)
    )
    return supported_models_base.ClipTarget(
      text_encoders.wan.WanT5Tokenizer, text_encoders.wan.te(**t5_detect)
    )


class WAN22_T2V(WAN21_T2V):
  unet_config = {
    "image_model": "wan2.1",
    "model_type": "t2v",
    "out_dim": 48,
  }

  latent_format = latent_formats.Wan22

  def get_model(self, state_dict, prefix="", device=None):
    out = model_base.WAN22(self, image_to_video=True, device=device)
    return out


models = [WAN22_T2V, WAN21_T2V]
