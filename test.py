from src.modules.image_to_video import ImageToVideo
from src.modules.clip_loader import CLIPLoader
from src.modules.unet_loader import UNetLoader
from src.modules.vae_loader import VAELoader
from src.modules.clip_text_encode import CLIPTextEncode
from src.modules.image_loader import ImageLoader
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

  image_loader = ImageLoader()
  image_rgb, _image_mask = image_loader.load_image("./cat.png")

  clip_loader = CLIPLoader()
  clip_model = clip_loader.load_clip(clip_path, "wan")

  vae_loader = VAELoader()
  [vae_model] = vae_loader.load_vae(vae_path)

  clip_text_encode = CLIPTextEncode()
  positive_conditioning = clip_text_encode.encode(clip_model, "a cat")
  negative_conditioning = clip_text_encode.encode(
    clip_model,
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  )

  image_to_video = ImageToVideo()
  positive, negative, out_latent = image_to_video.execute(
    positive=positive_conditioning,
    negative=negative_conditioning,
    vae=vae_model,
    width=512,
    height=512,
    length=16,
    batch_size=1,
    start_image=image_rgb,
  )

  print(out_latent)

  while True:
    time.sleep(1)


if __name__ == "__main__":
  # Setup logging
  logging.basicConfig(level=logging.DEBUG)
  main()
