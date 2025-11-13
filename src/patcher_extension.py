from typing import Callable


class WrappersMP:
  OUTER_SAMPLE = "outer_sample"
  PREPARE_SAMPLING = "prepare_sampling"
  SAMPLER_SAMPLE = "sampler_sample"
  PREDICT_NOISE = "predict_noise"
  CALC_COND_BATCH = "calc_cond_batch"
  APPLY_MODEL = "apply_model"
  DIFFUSION_MODEL = "diffusion_model"

  # wrappers dict is in the format:
  # {"wrapper_type": {"key": [Callable1, Callable2, ...]} }
  @classmethod
  def init_wrappers(cls) -> dict[str, dict[str, list[Callable]]]:
    return {}


def get_all_wrappers(
  wrapper_type: str, transformer_options: dict, is_model_options=False
):
  if is_model_options:
    transformer_options = transformer_options.get("transformer_options", {})
  w_list = []
  wrappers: dict[str, list] = transformer_options.get("wrappers", {})
  for w in wrappers.get(wrapper_type, {}).values():
    w_list.extend(w)
  return w_list


class WrapperExecutor:
  """Handles call stack of wrappers around a function in an ordered manner."""

  def __init__(
    self, original: Callable, class_obj: object, wrappers: list[Callable], idx: int
  ):
    # NOTE: class_obj exists so that wrappers surrounding a class method can access
    #       the class instance at runtime via executor.class_obj
    self.original = original
    self.class_obj = class_obj
    self.wrappers = wrappers.copy()
    self.idx = idx
    self.is_last = idx == len(wrappers)

  def __call__(self, *args, **kwargs):
    """Calls the next wrapper or original function, whichever is appropriate."""
    new_executor = self._create_next_executor()
    return new_executor.execute(*args, **kwargs)

  def execute(self, *args, **kwargs):
    """Used to initiate executor internally - DO NOT use this if you received executor in wrapper."""
    args = list(args)
    kwargs = dict(kwargs)
    if self.is_last:
      return self.original(*args, **kwargs)
    return self.wrappers[self.idx](self, *args, **kwargs)

  def _create_next_executor(self) -> "WrapperExecutor":
    new_idx = self.idx + 1
    if new_idx > len(self.wrappers):
      raise Exception(
        "Wrapper idx exceeded available wrappers; something went very wrong."
      )
    if self.class_obj is None:
      return WrapperExecutor.new_executor(self.original, self.wrappers, new_idx)
    return WrapperExecutor.new_class_executor(
      self.original, self.class_obj, self.wrappers, new_idx
    )

  @classmethod
  def new_executor(cls, original: Callable, wrappers: list[Callable], idx=0):
    return cls(original, class_obj=None, wrappers=wrappers, idx=idx)

  @classmethod
  def new_class_executor(
    cls, original: Callable, class_obj: object, wrappers: list[Callable], idx=0
  ):
    return cls(original, class_obj, wrappers, idx=idx)
