import src.utils as utils
from src.vae import VAE


class VAELoader:
  def load_vae(self, vae_path):
    sd = utils.load_torch_file(vae_path)
    vae = VAE(sd=sd)
    vae.throw_exception_if_invalid()
    return (vae,)
