import torch
import src.sd as sd


class CLIPLoader:
  def load_clip(self, clip_path: str, type="stable_diffusion", device="default"):
    clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.STABLE_DIFFUSION)

    model_options = {}
    if device == "cpu":
      model_options["load_device"] = model_options["offload_device"] = torch.device(
        "cpu"
      )

    # TODO: get embedding directory from config
    embedding_directory = ""

    clip = sd.load_clip(
      ckpt_paths=[clip_path],
      embedding_directory=embedding_directory,
      clip_type=clip_type,
      model_options=model_options,
    )
    return (clip,)
