from src.modules.clip_loader import CLIPLoader
from src.modules.unet_loader import UNetLoader
import logging
import time


def main():
  unet_path = "/home/ubuntu/comfyui-redis-worker/comfyui-models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
  clip_path = "/home/ubuntu/comfyui-redis-worker/comfyui-models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

  unet_loader = UNetLoader()
  wan_high_model = unet_loader.load_unet(unet_path)

  clip_loader = CLIPLoader()
  clip_model = clip_loader.load_clip(clip_path, "wan")

  while True:
    time.sleep(1)


if __name__ == "__main__":
  # Setup logging
  logging.basicConfig(level=logging.DEBUG)
  main()
