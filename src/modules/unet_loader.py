from src.sd import load_diffusion_model


class UNetLoader:
  def load_unet(self, unet_path):
    model = load_diffusion_model(unet_path, model_options={})
    return (model,)
