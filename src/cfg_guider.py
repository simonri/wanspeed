import torch

from src.model_patcher import ModelPatcher
import src.patcher_extension as patcher_extension
from src.samplers import (
  sampling_function,
  process_conds,
  cast_to_load_options,
  preprocess_conds_hooks,
  get_total_hook_groups_in_conds,
  filter_registered_hooks_on_conds,
)
import src.sampler_helpers as sampler_helpers
import src.model_patcher as model_patcher
import src.utils as utils
import src.nested_tensor as nested_tensor
import src.hooks as hooks


class CFGGuider:
  def __init__(self, model_patcher: ModelPatcher):
    self.model_patcher = model_patcher
    self.model_options = model_patcher.model_options
    self.original_conds = {}
    self.cfg = 1.0

  def set_conds(self, positive, negative):
    self.inner_set_conds({"positive": positive, "negative": negative})

  def set_cfg(self, cfg):
    self.cfg = cfg

  def inner_set_conds(self, conds):
    for k in conds:
      self.original_conds[k] = sampler_helpers.convert_cond(conds[k])

  def __call__(self, *args, **kwargs):
    return self.outer_predict_noise(*args, **kwargs)

  def outer_predict_noise(self, x, timestep, model_options={}, seed=None):
    return patcher_extension.WrapperExecutor.new_class_executor(
      self.predict_noise,
      self,
      patcher_extension.get_all_wrappers(
        patcher_extension.WrappersMP.PREDICT_NOISE,
        self.model_options,
        is_model_options=True,
      ),
    ).execute(x, timestep, model_options, seed)

  def predict_noise(self, x, timestep, model_options={}, seed=None):
    return sampling_function(
      self.inner_model,
      x,
      timestep,
      self.conds.get("negative", None),
      self.conds.get("positive", None),
      self.cfg,
      model_options=model_options,
      seed=seed,
    )

  def inner_sample(
    self,
    noise,
    latent_image,
    device,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes=None,
  ):
    if (
      latent_image is not None and torch.count_nonzero(latent_image) > 0
    ):  # Don't shift the empty latent image.
      latent_image = self.inner_model.process_latent_in(latent_image)

    self.conds = process_conds(
      self.inner_model,
      noise,
      self.conds,
      device,
      latent_image,
      denoise_mask,
      seed,
      latent_shapes=latent_shapes,
    )

    extra_model_options = model_patcher.create_model_options_clone(self.model_options)
    extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
    extra_args = {"model_options": extra_model_options, "seed": seed}

    executor = patcher_extension.WrapperExecutor.new_class_executor(
      sampler.sample,
      sampler,
      patcher_extension.get_all_wrappers(
        patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        extra_args["model_options"],
        is_model_options=True,
      ),
    )
    samples = executor.execute(
      self,
      sigmas,
      extra_args,
      callback,
      noise,
      latent_image,
      denoise_mask,
      disable_pbar,
    )
    return self.inner_model.process_latent_out(samples.to(torch.float32))

  def outer_sample(
    self,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
  ):
    self.inner_model, self.conds, self.loaded_models = sampler_helpers.prepare_sampling(
      self.model_patcher, noise.shape, self.conds, self.model_options
    )
    device = self.model_patcher.load_device

    if denoise_mask is not None:
      denoise_mask = sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    sigmas = sigmas.to(device)
    cast_to_load_options(
      self.model_options, device=device, dtype=self.model_patcher.model_dtype()
    )

    try:
      self.model_patcher.pre_run()
      output = self.inner_sample(
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=latent_shapes,
      )
    finally:
      self.model_patcher.cleanup()

    sampler_helpers.cleanup_models(self.conds, self.loaded_models)
    del self.inner_model
    del self.loaded_models
    return output

  def sample(
    self,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
  ):
    if sigmas.shape[-1] == 0:
      return latent_image

    if latent_image.is_nested:
      latent_image, latent_shapes = utils.pack_latents(latent_image.unbind())
      noise, _ = utils.pack_latents(noise.unbind())
    else:
      latent_shapes = [latent_image.shape]

    self.conds = {}
    for k in self.original_conds:
      self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
    preprocess_conds_hooks(self.conds)

    try:
      orig_model_options = self.model_options
      self.model_options = model_patcher.create_model_options_clone(self.model_options)
      # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
      orig_hook_mode = self.model_patcher.hook_mode
      if get_total_hook_groups_in_conds(self.conds) <= 1:
        self.model_patcher.hook_mode = hooks.EnumHookMode.MinVram
      sampler_helpers.prepare_model_patcher(
        self.model_patcher, self.conds, self.model_options
      )
      filter_registered_hooks_on_conds(self.conds, self.model_options)
      executor = patcher_extension.WrapperExecutor.new_class_executor(
        self.outer_sample,
        self,
        patcher_extension.get_all_wrappers(
          patcher_extension.WrappersMP.OUTER_SAMPLE,
          self.model_options,
          is_model_options=True,
        ),
      )
      output = executor.execute(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=latent_shapes,
      )
    finally:
      cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
      self.model_options = orig_model_options
      self.model_patcher.hook_mode = orig_hook_mode
      self.model_patcher.restore_hook_patches()

    del self.conds

    if len(latent_shapes) > 1:
      output = nested_tensor.NestedTensor(utils.unpack_latents(output, latent_shapes))
    return output
