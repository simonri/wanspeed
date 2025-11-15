from PIL import Image, ImageOps
import numpy as np
import torch


class ImageLoader:
  def load_image(self, image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.resize((480, 480))

    if "A" in img.getbands():
      # Image has alpha
      alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
      mask = 1.0 - torch.from_numpy(alpha)  # invert to match SD mask convention
    else:
      # No alpha -> zero mask
      w, h = img.size
      mask = torch.zeros((h, w), dtype=torch.float32)

    rgb = img.convert("RGB")
    rgb = np.array(rgb).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb)[None, ...]

    mask = mask.unsqueeze(0)
    return rgb, mask
