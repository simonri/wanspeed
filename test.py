from src.modules.clip_loader import CLIPLoader
from src.modules.unet_loader import UNetLoader
from src.modules.vae_loader import VAELoader
from src.modules.clip_text_encode import CLIPTextEncode
import logging
import time


def main():
  unet_path = "/home/ubuntu/comfyui-redis-worker/comfyui-models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
  clip_path = "/home/ubuntu/comfyui-redis-worker/comfyui-models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  vae_path = (
    "/home/ubuntu/comfyui-redis-worker/comfyui-models/vae/wan_2.1_vae.safetensors"
  )

  # unet_loader = UNetLoader()
  # wan_high_model = unet_loader.load_unet(unet_path)

  clip_loader = CLIPLoader()
  clip_model = clip_loader.load_clip(clip_path, "wan")

  # vae_loader = VAELoader()
  # vae_model = vae_loader.load_vae(vae_path)

  clip_text_encode = CLIPTextEncode()
  clip_text_encoded = clip_text_encode.encode(clip_model, "a cat")

  print(clip_text_encoded)

  while True:
    time.sleep(1)


if __name__ == "__main__":
  # Setup logging
  logging.basicConfig(level=logging.DEBUG)
  main()
