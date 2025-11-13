import torch
from src.wan.model_base import WAN22


class SimpleModelConfig:
  def __init__(self):
    # Basic WAN22 configuration based on WanModel parameters
    self.unet_config = {
      "model_type": "t2v",
      "patch_size": (1, 2, 2),
      "text_len": 1,
      "in_dim": 16,
      "dim": 5120,
      "ffn_dim": 12800,
      "freq_dim": 256,
      "text_dim": 4096,
      "out_dim": 16,
      "num_heads": 20,
      "num_layers": 40,
      "window_size": (-1, -1),
      "qk_norm": True,
      "cross_attn_norm": True,
      "eps": 1e-6,
      "dtype": torch.float16,
    }

    # Latent format (simplified)
    class SimpleLatentFormat:
      def __init__(self):
        self.latent_channels = 16

      def process_in(self, latent):
        return latent

      def process_out(self, latent):
        return latent

    self.latent_format = SimpleLatentFormat()
    self.manual_cast_dtype = torch.float16
    self.custom_operations = None
    self.scaled_fp8 = None
    self.optimizations = {"fp8": False}
    self.memory_usage_factor = 1.0


def main():
  # Create model configuration
  model_config = SimpleModelConfig()

  # Create WAN22 model
  model = WAN22(model_config)

  print("WAN22 model created successfully!")
  print(f"Model type: {model.model_type}")
  print(f"Device: {model.device}")
  print(f"Manual cast dtype: {model.manual_cast_dtype}")


if __name__ == "__main__":
  main()
