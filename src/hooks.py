import enum


class EnumHookMode(enum.Enum):
  """
  Priority of hook memory optimization vs. speed, mostly related to WeightHooks.

  MinVram: No caching will occur for any operations related to hooks.
  MaxSpeed: Excess VRAM (and RAM, once VRAM is sufficiently depleted) will be used to cache hook weights when switching hook groups.
  """

  MinVram = "minvram"
  MaxSpeed = "maxspeed"
