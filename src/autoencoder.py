import torch
import logging
from typing import Any, Dict, Union, Tuple
from contextlib import contextmanager
from src.utils import get_obj_from_str, instantiate_from_config
from src.model_management import LitEma

class AbstractAutoencoder(torch.nn.Module):
  """
  This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
  unCLIP models, etc. Hence, it is fairly general, and specific features
  (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
  """

  def __init__(
    self,
    ema_decay: Union[None, float] = None,
    monitor: Union[None, str] = None,
    input_key: str = "jpg",
    **kwargs,
  ):
    super().__init__()

    self.input_key = input_key
    self.use_ema = ema_decay is not None
    if monitor is not None:
      self.monitor = monitor

    if self.use_ema:
      self.model_ema = LitEma(self, decay=ema_decay)
      logging.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

  def get_input(self, batch) -> Any:
    raise NotImplementedError()

  def on_train_batch_end(self, *args, **kwargs):
    # for EMA computation
    if self.use_ema:
      self.model_ema(self)

  @contextmanager
  def ema_scope(self, context=None):
    if self.use_ema:
      self.model_ema.store(self.parameters())
      self.model_ema.copy_to(self)
      if context is not None:
        logging.info(f"{context}: Switched to EMA weights")
    try:
      yield None
    finally:
      if self.use_ema:
        self.model_ema.restore(self.parameters())
        if context is not None:
          logging.info(f"{context}: Restored training weights")

  def encode(self, *args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("encode()-method of abstract base class called")

  def decode(self, *args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("decode()-method of abstract base class called")

  def instantiate_optimizer_from_config(self, params, lr, cfg):
    logging.info(f"loading >>> {cfg['target']} <<< optimizer from config")
    return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))

  def configure_optimizers(self) -> Any:
    raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
  """
  Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
  (we also restore them explicitly as special cases for legacy reasons).
  Regularizations such as KL or VQ are moved to the regularizer class.
  """

  def __init__(
    self,
    *args,
    encoder_config: Dict,
    decoder_config: Dict,
    regularizer_config: Dict,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)

    self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
    self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
    self.regularization = instantiate_from_config(regularizer_config)

  def get_last_layer(self):
    return self.decoder.get_last_layer()

  def encode(
    self,
    x: torch.Tensor,
    return_reg_log: bool = False,
    unregularized: bool = False,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    z = self.encoder(x)
    if unregularized:
      return z, dict()
    z, reg_log = self.regularization(z)
    if return_reg_log:
      return z, reg_log
    return z

  def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
    x = self.decoder(z, **kwargs)
    return x

  def forward(
    self, x: torch.Tensor, **additional_decode_kwargs
  ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    z, reg_log = self.encode(x, return_reg_log=True)
    dec = self.decode(z, **additional_decode_kwargs)
    return z, dec, reg_log
