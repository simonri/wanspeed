import torch
import src.utils as utils
import src.modules.helpers as helpers


class ImageToVideo:
  def execute(
    cls,
    positive,
    negative,
    vae,
    width,
    height,
    length,
    batch_size,
    start_image=None,
    clip_vision_output=None,
  ):
    latent = torch.zeros(
      [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
      device=torch.device("cpu"),
    )
    if start_image is not None:
      start_image = utils.common_upscale(
        start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
      ).movedim(1, -1)
      image = (
        torch.ones(
          (length, height, width, start_image.shape[-1]),
          device=start_image.device,
          dtype=start_image.dtype,
        )
        * 0.5
      )
      image[: start_image.shape[0]] = start_image

      concat_latent_image = vae.encode(image[:, :, :, :3])
      mask = torch.ones(
        (
          1,
          1,
          latent.shape[2],
          concat_latent_image.shape[-2],
          concat_latent_image.shape[-1],
        ),
        device=start_image.device,
        dtype=start_image.dtype,
      )
      mask[:, :, : ((start_image.shape[0] - 1) // 4) + 1] = 0.0

      positive = helpers.conditioning_set_values(
        positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
      )
      negative = helpers.conditioning_set_values(
        negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
      )

    if clip_vision_output is not None:
      positive = helpers.conditioning_set_values(
        positive, {"clip_vision_output": clip_vision_output}
      )
      negative = helpers.conditioning_set_values(
        negative, {"clip_vision_output": clip_vision_output}
      )

    out_latent = {}
    out_latent["samples"] = latent
    return positive, negative, out_latent
