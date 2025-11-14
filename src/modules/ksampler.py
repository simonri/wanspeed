import src.sample as sample
import torch


def common_ksampler(
  model,
  seed,
  steps,
  cfg,
  sampler_name,
  scheduler,
  positive,
  negative,
  latent,
  denoise=1.0,
  disable_noise=False,
  start_step=None,
  last_step=None,
  force_full_denoise=False,
):
  latent_image = latent["samples"]
  latent_image = sample.fix_empty_latent_channels(model, latent_image)

  if disable_noise:
    noise = torch.zeros(
      latent_image.size(),
      dtype=latent_image.dtype,
      layout=latent_image.layout,
      device="cpu",
    )
  else:
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = sample.prepare_noise(latent_image, seed, batch_inds)

  noise_mask = None
  if "noise_mask" in latent:
    noise_mask = latent["noise_mask"]

  samples = sample.sample(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=denoise,
    disable_noise=disable_noise,
    start_step=start_step,
    last_step=last_step,
    force_full_denoise=force_full_denoise,
    noise_mask=noise_mask,
    callback=None,
    disable_pbar=True,
    seed=seed,
  )
  out = latent.copy()
  out["samples"] = samples
  return (out,)


class KSamplerAdvanced:
  def sample(
    self,
    model,
    add_noise,
    noise_seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    start_at_step,
    end_at_step,
    return_with_leftover_noise,
    denoise=1.0,
  ):
    force_full_denoise = True
    if return_with_leftover_noise == "enable":
      force_full_denoise = False
    disable_noise = False
    if add_noise == "disable":
      disable_noise = True
    return common_ksampler(
      model,
      noise_seed,
      steps,
      cfg,
      sampler_name,
      scheduler,
      positive,
      negative,
      latent_image,
      denoise=denoise,
      disable_noise=disable_noise,
      start_step=start_at_step,
      last_step=end_at_step,
      force_full_denoise=force_full_denoise,
    )
