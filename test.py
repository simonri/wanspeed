from src.modules.unet_loader import UNetLoader
import logging


def main():
  unet_path = "/home/ubuntu/comfyui-redis-worker/comfyui-models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"

  loader = UNetLoader()
  model = loader.load_unet(unet_path)


if __name__ == "__main__":
  # Setup logging
  logging.basicConfig(level=logging.DEBUG)
  main()
