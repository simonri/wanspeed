import src.model_management as model_management
import src.diffusers_convert as diffusers_convert
from src.autoencoder import AutoencodingEngine
import torch
import math
import json


class VAE:
  def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
    if "decoder.up_blocks.0.resnets.0.norm1.weight" in sd.keys():  # diffusers format
      sd = diffusers_convert.convert_vae_state_dict(sd)

    if model_management.is_amd():
      VAE_KL_MEM_RATIO = 2.73
    else:
      VAE_KL_MEM_RATIO = 1.0

    self.memory_used_encode = (
      lambda shape, dtype: (1767 * shape[2] * shape[3])
      * model_management.dtype_size(dtype)
      * VAE_KL_MEM_RATIO
    )  # These are for AutoencoderKL and need tweaking (should be lower)
    self.memory_used_decode = (
      lambda shape, dtype: (2178 * shape[2] * shape[3] * 64)
      * model_management.dtype_size(dtype)
      * VAE_KL_MEM_RATIO
    )
    self.downscale_ratio = 8
    self.upscale_ratio = 8
    self.latent_channels = 4
    self.latent_dim = 2
    self.output_channels = 3
    self.process_input = lambda image: image * 2.0 - 1.0
    self.process_output = lambda image: torch.clamp(
      (image + 1.0) / 2.0, min=0.0, max=1.0
    )
    self.working_dtypes = [torch.bfloat16, torch.float32]
    self.disable_offload = False
    self.not_video = False
    self.size = None

    self.downscale_index_formula = None
    self.upscale_index_formula = None
    self.extra_1d_channel = None
    self.crop_input = True

    if config is None:
      if "decoder.mid.block_1.mix_factor" in sd:
        encoder_config = {
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult": [1, 2, 4, 4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        }
        decoder_config = encoder_config.copy()
        decoder_config["video_kernel_size"] = [3, 1, 1]
        decoder_config["alpha"] = 0.0
        self.first_stage_model = AutoencodingEngine(
          regularizer_config={
            "target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"
          },
          encoder_config={
            "target": "comfy.ldm.modules.diffusionmodules.model.Encoder",
            "params": encoder_config,
          },
          decoder_config={
            "target": "comfy.ldm.modules.temporal_ae.VideoDecoder",
            "params": decoder_config,
          },
        )
      elif "taesd_decoder.1.weight" in sd:
        self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
        self.first_stage_model = comfy.taesd.taesd.TAESD(
          latent_channels=self.latent_channels
        )
      elif "vquantizer.codebook.weight" in sd:  # VQGan: stage a of stable cascade
        self.first_stage_model = StageA()
        self.downscale_ratio = 4
        self.upscale_ratio = 4
        # TODO
        # self.memory_used_encode
        # self.memory_used_decode
        self.process_input = lambda image: image
        self.process_output = lambda image: image
      elif (
        "backbone.1.0.block.0.1.num_batches_tracked" in sd
      ):  # effnet: encoder for stage c latent of stable cascade
        self.first_stage_model = StageC_coder()
        self.downscale_ratio = 32
        self.latent_channels = 16
        new_sd = {}
        for k in sd:
          new_sd["encoder.{}".format(k)] = sd[k]
        sd = new_sd
      elif (
        "blocks.11.num_batches_tracked" in sd
      ):  # previewer: decoder for stage c latent of stable cascade
        self.first_stage_model = StageC_coder()
        self.latent_channels = 16
        new_sd = {}
        for k in sd:
          new_sd["previewer.{}".format(k)] = sd[k]
        sd = new_sd
      elif (
        "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd
      ):  # combined effnet and previewer for stable cascade
        self.first_stage_model = StageC_coder()
        self.downscale_ratio = 32
        self.latent_channels = 16
      elif "decoder.conv_in.weight" in sd:
        if sd["decoder.conv_in.weight"].shape[1] == 64:
          ddconfig = {
            "block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "in_channels": 3,
            "out_channels": 3,
            "num_res_blocks": 2,
            "ffactor_spatial": 32,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
          }
          self.latent_channels = ddconfig["z_channels"] = sd[
            "decoder.conv_in.weight"
          ].shape[1]
          self.downscale_ratio = 32
          self.upscale_ratio = 32
          self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
          self.first_stage_model = AutoencodingEngine(
            regularizer_config={
              "target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"
            },
            encoder_config={
              "target": "comfy.ldm.hunyuan_video.vae.Encoder",
              "params": ddconfig,
            },
            decoder_config={
              "target": "comfy.ldm.hunyuan_video.vae.Decoder",
              "params": ddconfig,
            },
          )

          self.memory_used_encode = lambda shape, dtype: (
            700 * shape[2] * shape[3]
          ) * model_management.dtype_size(dtype)
          self.memory_used_decode = lambda shape, dtype: (
            700 * shape[2] * shape[3] * 32 * 32
          ) * model_management.dtype_size(dtype)
        elif sd["decoder.conv_in.weight"].shape[1] == 32:
          ddconfig = {
            "block_out_channels": [128, 256, 512, 1024, 1024],
            "in_channels": 3,
            "out_channels": 3,
            "num_res_blocks": 2,
            "ffactor_spatial": 16,
            "ffactor_temporal": 4,
            "downsample_match_channel": True,
            "upsample_match_channel": True,
            "refiner_vae": False,
          }
          self.latent_channels = ddconfig["z_channels"] = sd[
            "decoder.conv_in.weight"
          ].shape[1]
          self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
          self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
          self.upscale_index_formula = (4, 16, 16)
          self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
          self.downscale_index_formula = (4, 16, 16)
          self.latent_dim = 3
          self.not_video = True
          self.first_stage_model = AutoencodingEngine(
            regularizer_config={
              "target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"
            },
            encoder_config={
              "target": "comfy.ldm.hunyuan_video.vae_refiner.Encoder",
              "params": ddconfig,
            },
            decoder_config={
              "target": "comfy.ldm.hunyuan_video.vae_refiner.Decoder",
              "params": ddconfig,
            },
          )

          self.memory_used_encode = lambda shape, dtype: (
            2800 * shape[-2] * shape[-1]
          ) * model_management.dtype_size(dtype)
          self.memory_used_decode = lambda shape, dtype: (
            2800 * shape[-3] * shape[-2] * shape[-1] * 16 * 16
          ) * model_management.dtype_size(dtype)
        else:
          # default SD1.x/SD2.x VAE parameters
          ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
          }

          if (
            "encoder.down.2.downsample.conv.weight" not in sd
            and "decoder.up.3.upsample.conv.weight" not in sd
          ):  # Stable diffusion x4 upscaler VAE
            ddconfig["ch_mult"] = [1, 2, 4]
            self.downscale_ratio = 4
            self.upscale_ratio = 4

          self.latent_channels = ddconfig["z_channels"] = sd[
            "decoder.conv_in.weight"
          ].shape[1]
          if "post_quant_conv.weight" in sd:
            self.first_stage_model = AutoencoderKL(
              ddconfig=ddconfig, embed_dim=sd["post_quant_conv.weight"].shape[1]
            )
          else:
            self.first_stage_model = AutoencodingEngine(
              regularizer_config={
                "target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"
              },
              encoder_config={
                "target": "comfy.ldm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
              },
              decoder_config={
                "target": "comfy.ldm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
              },
            )
      elif "decoder.layers.1.layers.0.beta" in sd:
        self.first_stage_model = AudioOobleckVAE()
        self.memory_used_encode = lambda shape, dtype: (
          1000 * shape[2]
        ) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (
          1000 * shape[2] * 2048
        ) * model_management.dtype_size(dtype)
        self.latent_channels = 64
        self.output_channels = 2
        self.upscale_ratio = 2048
        self.downscale_ratio = 2048
        self.latent_dim = 1
        self.process_output = lambda audio: audio
        self.process_input = lambda audio: audio
        self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        self.disable_offload = True
      elif (
        "blocks.2.blocks.3.stack.5.weight" in sd
        or "decoder.blocks.2.blocks.3.stack.5.weight" in sd
        or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd
        or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd
      ):  # genmo mochi vae
        if "blocks.2.blocks.3.stack.5.weight" in sd:
          sd = comfy.utils.state_dict_prefix_replace(sd, {"": "decoder."})
        if "layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
          sd = comfy.utils.state_dict_prefix_replace(sd, {"": "encoder."})
        self.first_stage_model = comfy.ldm.genmo.vae.model.VideoVAE()
        self.latent_channels = 12
        self.latent_dim = 3
        self.memory_used_decode = lambda shape, dtype: (
          1000 * shape[2] * shape[3] * shape[4] * (6 * 8 * 8)
        ) * model_management.dtype_size(dtype)
        self.memory_used_encode = lambda shape, dtype: (
          1.5 * max(shape[2], 7) * shape[3] * shape[4] * (6 * 8 * 8)
        ) * model_management.dtype_size(dtype)
        self.upscale_ratio = (lambda a: max(0, a * 6 - 5), 8, 8)
        self.upscale_index_formula = (6, 8, 8)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 5) / 6)), 8, 8)
        self.downscale_index_formula = (6, 8, 8)
        self.working_dtypes = [torch.float16, torch.float32]
      elif (
        "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd
      ):  # lightricks ltxv
        tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
        version = 0
        if tensor_conv1.shape[0] == 512:
          version = 0
        elif tensor_conv1.shape[0] == 1024:
          version = 1
          if "encoder.down_blocks.1.conv.conv.bias" in sd:
            version = 2
        vae_config = None
        if metadata is not None and "config" in metadata:
          vae_config = json.loads(metadata["config"]).get("vae", None)
        self.first_stage_model = (
          comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE(
            version=version, config=vae_config
          )
        )
        self.latent_channels = 128
        self.latent_dim = 3
        self.memory_used_decode = lambda shape, dtype: (
          900 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)
        ) * model_management.dtype_size(dtype)
        self.memory_used_encode = lambda shape, dtype: (
          70 * max(shape[2], 7) * shape[3] * shape[4]
        ) * model_management.dtype_size(dtype)
        self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
        self.upscale_index_formula = (8, 32, 32)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
        self.downscale_index_formula = (8, 32, 32)
        self.working_dtypes = [torch.bfloat16, torch.float32]
      elif (
        "decoder.conv_in.conv.weight" in sd
        and sd["decoder.conv_in.conv.weight"].shape[1] == 32
      ):
        ddconfig = {
          "block_out_channels": [128, 256, 512, 1024, 1024],
          "in_channels": 3,
          "out_channels": 3,
          "num_res_blocks": 2,
          "ffactor_spatial": 16,
          "ffactor_temporal": 4,
          "downsample_match_channel": True,
          "upsample_match_channel": True,
        }
        ddconfig["z_channels"] = sd["decoder.conv_in.conv.weight"].shape[1]
        self.latent_channels = 64
        self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
        self.upscale_index_formula = (4, 16, 16)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
        self.downscale_index_formula = (4, 16, 16)
        self.latent_dim = 3
        self.not_video = True
        self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        self.first_stage_model = AutoencodingEngine(
          regularizer_config={
            "target": "comfy.ldm.models.autoencoder.EmptyRegularizer"
          },
          encoder_config={
            "target": "comfy.ldm.hunyuan_video.vae_refiner.Encoder",
            "params": ddconfig,
          },
          decoder_config={
            "target": "comfy.ldm.hunyuan_video.vae_refiner.Decoder",
            "params": ddconfig,
          },
        )

        self.memory_used_encode = lambda shape, dtype: (
          1400 * shape[-2] * shape[-1]
        ) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (
          1400 * shape[-3] * shape[-2] * shape[-1] * 16 * 16
        ) * model_management.dtype_size(dtype)
      elif "decoder.conv_in.conv.weight" in sd:
        ddconfig = {
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult": [1, 2, 4, 4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        }
        ddconfig["conv3d"] = True
        ddconfig["time_compress"] = 4
        self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
        self.upscale_index_formula = (4, 8, 8)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
        self.downscale_index_formula = (4, 8, 8)
        self.latent_dim = 3
        self.latent_channels = ddconfig["z_channels"] = sd[
          "decoder.conv_in.conv.weight"
        ].shape[1]
        self.first_stage_model = AutoencoderKL(
          ddconfig=ddconfig, embed_dim=sd["post_quant_conv.weight"].shape[1]
        )
        self.memory_used_decode = lambda shape, dtype: (
          1500 * shape[2] * shape[3] * shape[4] * (4 * 8 * 8)
        ) * model_management.dtype_size(dtype)
        self.memory_used_encode = lambda shape, dtype: (
          900 * max(shape[2], 2) * shape[3] * shape[4]
        ) * model_management.dtype_size(dtype)
        self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
      elif "decoder.unpatcher3d.wavelets" in sd:
        self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 8, 8)
        self.upscale_index_formula = (8, 8, 8)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 8, 8)
        self.downscale_index_formula = (8, 8, 8)
        self.latent_dim = 3
        self.latent_channels = 16
        ddconfig = {
          "z_channels": 16,
          "latent_channels": self.latent_channels,
          "z_factor": 1,
          "resolution": 1024,
          "in_channels": 3,
          "out_channels": 3,
          "channels": 128,
          "channels_mult": [2, 4, 4],
          "num_res_blocks": 2,
          "attn_resolutions": [32],
          "dropout": 0.0,
          "patch_size": 4,
          "num_groups": 1,
          "temporal_compression": 8,
          "spacial_compression": 8,
        }
        self.first_stage_model = comfy.ldm.cosmos.vae.CausalContinuousVideoTokenizer(
          **ddconfig
        )
        # TODO: these values are a bit off because this is not a standard VAE
        self.memory_used_decode = lambda shape, dtype: (
          50 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)
        ) * model_management.dtype_size(dtype)
        self.memory_used_encode = lambda shape, dtype: (
          50 * (round((shape[2] + 7) / 8) * 8) * shape[3] * shape[4]
        ) * model_management.dtype_size(dtype)
        self.working_dtypes = [torch.bfloat16, torch.float32]
      elif "decoder.middle.0.residual.0.gamma" in sd:
        if "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd:  # Wan 2.2 VAE
          self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
          self.upscale_index_formula = (4, 16, 16)
          self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
          self.downscale_index_formula = (4, 16, 16)
          self.latent_dim = 3
          self.latent_channels = 48
          ddconfig = {
            "dim": 160,
            "z_dim": self.latent_channels,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
          }
          self.first_stage_model = comfy.ldm.wan.vae2_2.WanVAE(**ddconfig)
          self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
          self.memory_used_encode = (
            lambda shape, dtype: 3300
            * shape[3]
            * shape[4]
            * model_management.dtype_size(dtype)
          )
          self.memory_used_decode = (
            lambda shape, dtype: 8000
            * shape[3]
            * shape[4]
            * (16 * 16)
            * model_management.dtype_size(dtype)
          )
        else:  # Wan 2.1 VAE
          self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
          self.upscale_index_formula = (4, 8, 8)
          self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
          self.downscale_index_formula = (4, 8, 8)
          self.latent_dim = 3
          self.latent_channels = 16
          ddconfig = {
            "dim": 96,
            "z_dim": self.latent_channels,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
          }
          self.first_stage_model = comfy.ldm.wan.vae.WanVAE(**ddconfig)
          self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
          self.memory_used_encode = (
            lambda shape, dtype: 6000
            * shape[3]
            * shape[4]
            * model_management.dtype_size(dtype)
          )
          self.memory_used_decode = (
            lambda shape, dtype: 7000
            * shape[3]
            * shape[4]
            * (8 * 8)
            * model_management.dtype_size(dtype)
          )
      # Hunyuan 3d v2 2.0 & 2.1
      elif "geo_decoder.cross_attn_decoder.ln_1.bias" in sd:
        self.latent_dim = 1

        def estimate_memory(shape, dtype, num_layers=16, kv_cache_multiplier=2):
          batch, num_tokens, hidden_dim = shape
          dtype_size = model_management.dtype_size(dtype)

          total_mem = (
            batch
            * num_tokens
            * hidden_dim
            * dtype_size
            * (1 + kv_cache_multiplier * num_layers)
          )
          return total_mem

        # better memory estimations
        self.memory_used_encode = (
          lambda shape, dtype, num_layers=8, kv_cache_multiplier=0: estimate_memory(
            shape, dtype, num_layers, kv_cache_multiplier
          )
        )

        self.memory_used_decode = (
          lambda shape, dtype, num_layers=16, kv_cache_multiplier=2: estimate_memory(
            shape, dtype, num_layers, kv_cache_multiplier
          )
        )

        self.first_stage_model = comfy.ldm.hunyuan3d.vae.ShapeVAE()
        self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]

      elif "vocoder.backbone.channel_layers.0.0.bias" in sd:  # Ace Step Audio
        self.first_stage_model = comfy.ldm.ace.vae.music_dcae_pipeline.MusicDCAE(
          source_sample_rate=44100
        )
        self.memory_used_encode = lambda shape, dtype: (
          shape[2] * 330
        ) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (
          shape[2] * shape[3] * 87000
        ) * model_management.dtype_size(dtype)
        self.latent_channels = 8
        self.output_channels = 2
        self.upscale_ratio = 4096
        self.downscale_ratio = 4096
        self.latent_dim = 2
        self.process_output = lambda audio: audio
        self.process_input = lambda audio: audio
        self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        self.disable_offload = True
        self.extra_1d_channel = 16
      elif "pixel_space_vae" in sd:
        self.first_stage_model = comfy.pixel_space_convert.PixelspaceConversionVAE()
        self.memory_used_encode = lambda shape, dtype: (
          1 * shape[2] * shape[3]
        ) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (
          1 * shape[2] * shape[3]
        ) * model_management.dtype_size(dtype)
        self.downscale_ratio = 1
        self.upscale_ratio = 1
        self.latent_channels = 3
        self.latent_dim = 2
        self.output_channels = 3
      elif "vocoder.activation_post.downsample.lowpass.filter" in sd:  # MMAudio VAE
        sample_rate = 16000
        if sample_rate == 16000:
          mode = "16k"
        else:
          mode = "44k"

        self.first_stage_model = comfy.ldm.mmaudio.vae.autoencoder.AudioAutoencoder(
          mode=mode
        )
        self.memory_used_encode = lambda shape, dtype: (
          30 * shape[2]
        ) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (
          90 * shape[2] * 1411.2
        ) * model_management.dtype_size(dtype)
        self.latent_channels = 20
        self.output_channels = 2
        self.upscale_ratio = 512 * (44100 / sample_rate)
        self.downscale_ratio = 512 * (44100 / sample_rate)
        self.latent_dim = 1
        self.process_output = lambda audio: audio
        self.process_input = lambda audio: audio
        self.working_dtypes = [torch.float32]
        self.crop_input = False
      else:
        logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
        self.first_stage_model = None
        return
    else:
      self.first_stage_model = AutoencoderKL(**(config["params"]))
    self.first_stage_model = self.first_stage_model.eval()

    m, u = self.first_stage_model.load_state_dict(sd, strict=False)
    if len(m) > 0:
      logging.warning("Missing VAE keys {}".format(m))

    if len(u) > 0:
      logging.debug("Leftover VAE keys {}".format(u))

    if device is None:
      device = model_management.vae_device()
    self.device = device
    offload_device = model_management.vae_offload_device()
    if dtype is None:
      dtype = model_management.vae_dtype(self.device, self.working_dtypes)
    self.vae_dtype = dtype
    self.first_stage_model.to(self.vae_dtype)
    self.output_device = model_management.intermediate_device()

    self.patcher = comfy.model_patcher.ModelPatcher(
      self.first_stage_model, load_device=self.device, offload_device=offload_device
    )
    logging.info(
      "VAE load device: {}, offload device: {}, dtype: {}".format(
        self.device, offload_device, self.vae_dtype
      )
    )
    self.model_size()

  def model_size(self):
    if self.size is not None:
      return self.size
    self.size = comfy.model_management.module_size(self.first_stage_model)
    return self.size

  def get_ram_usage(self):
    return self.model_size()

  def throw_exception_if_invalid(self):
    if self.first_stage_model is None:
      raise RuntimeError(
        "ERROR: VAE is invalid: None\n\nIf the VAE is from a checkpoint loader node your checkpoint does not contain a valid VAE."
      )

  def vae_encode_crop_pixels(self, pixels):
    if not self.crop_input:
      return pixels

    downscale_ratio = self.spacial_compression_encode()

    dims = pixels.shape[1:-1]
    for d in range(len(dims)):
      x = (dims[d] // downscale_ratio) * downscale_ratio
      x_offset = (dims[d] % downscale_ratio) // 2
      if x != dims[d]:
        pixels = pixels.narrow(d + 1, x_offset, x)
    return pixels

  def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
    steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(
      samples.shape[3], samples.shape[2], tile_x, tile_y, overlap
    )
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(
      samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap
    )
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(
      samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap
    )
    pbar = comfy.utils.ProgressBar(steps)

    decode_fn = lambda a: self.first_stage_model.decode(
      a.to(self.vae_dtype).to(self.device)
    ).float()
    output = self.process_output(
      (
        comfy.utils.tiled_scale(
          samples,
          decode_fn,
          tile_x // 2,
          tile_y * 2,
          overlap,
          upscale_amount=self.upscale_ratio,
          output_device=self.output_device,
          pbar=pbar,
        )
        + comfy.utils.tiled_scale(
          samples,
          decode_fn,
          tile_x * 2,
          tile_y // 2,
          overlap,
          upscale_amount=self.upscale_ratio,
          output_device=self.output_device,
          pbar=pbar,
        )
        + comfy.utils.tiled_scale(
          samples,
          decode_fn,
          tile_x,
          tile_y,
          overlap,
          upscale_amount=self.upscale_ratio,
          output_device=self.output_device,
          pbar=pbar,
        )
      )
      / 3.0
    )
    return output

  def decode_tiled_1d(self, samples, tile_x=128, overlap=32):
    if samples.ndim == 3:
      decode_fn = lambda a: self.first_stage_model.decode(
        a.to(self.vae_dtype).to(self.device)
      ).float()
    else:
      og_shape = samples.shape
      samples = samples.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
      decode_fn = lambda a: self.first_stage_model.decode(
        a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1]))
        .to(self.vae_dtype)
        .to(self.device)
      ).float()

    return self.process_output(
      comfy.utils.tiled_scale_multidim(
        samples,
        decode_fn,
        tile=(tile_x,),
        overlap=overlap,
        upscale_amount=self.upscale_ratio,
        out_channels=self.output_channels,
        output_device=self.output_device,
      )
    )

  def decode_tiled_3d(
    self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)
  ):
    decode_fn = lambda a: self.first_stage_model.decode(
      a.to(self.vae_dtype).to(self.device)
    ).float()
    return self.process_output(
      comfy.utils.tiled_scale_multidim(
        samples,
        decode_fn,
        tile=(tile_t, tile_x, tile_y),
        overlap=overlap,
        upscale_amount=self.upscale_ratio,
        out_channels=self.output_channels,
        index_formulas=self.upscale_index_formula,
        output_device=self.output_device,
      )
    )

  def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
    steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(
      pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap
    )
    steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(
      pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap
    )
    steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(
      pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap
    )
    pbar = comfy.utils.ProgressBar(steps)

    encode_fn = lambda a: self.first_stage_model.encode(
      (self.process_input(a)).to(self.vae_dtype).to(self.device)
    ).float()
    samples = comfy.utils.tiled_scale(
      pixel_samples,
      encode_fn,
      tile_x,
      tile_y,
      overlap,
      upscale_amount=(1 / self.downscale_ratio),
      out_channels=self.latent_channels,
      output_device=self.output_device,
      pbar=pbar,
    )
    samples += comfy.utils.tiled_scale(
      pixel_samples,
      encode_fn,
      tile_x * 2,
      tile_y // 2,
      overlap,
      upscale_amount=(1 / self.downscale_ratio),
      out_channels=self.latent_channels,
      output_device=self.output_device,
      pbar=pbar,
    )
    samples += comfy.utils.tiled_scale(
      pixel_samples,
      encode_fn,
      tile_x // 2,
      tile_y * 2,
      overlap,
      upscale_amount=(1 / self.downscale_ratio),
      out_channels=self.latent_channels,
      output_device=self.output_device,
      pbar=pbar,
    )
    samples /= 3.0
    return samples

  def encode_tiled_1d(self, samples, tile_x=256 * 2048, overlap=64 * 2048):
    if self.latent_dim == 1:
      encode_fn = lambda a: self.first_stage_model.encode(
        (self.process_input(a)).to(self.vae_dtype).to(self.device)
      ).float()
      out_channels = self.latent_channels
      upscale_amount = 1 / self.downscale_ratio
    else:
      extra_channel_size = self.extra_1d_channel
      out_channels = self.latent_channels * extra_channel_size
      tile_x = tile_x // extra_channel_size
      overlap = overlap // extra_channel_size
      upscale_amount = 1 / self.downscale_ratio
      encode_fn = (
        lambda a: self.first_stage_model.encode(
          (self.process_input(a)).to(self.vae_dtype).to(self.device)
        )
        .reshape(1, out_channels, -1)
        .float()
      )

    out = comfy.utils.tiled_scale_multidim(
      samples,
      encode_fn,
      tile=(tile_x,),
      overlap=overlap,
      upscale_amount=upscale_amount,
      out_channels=out_channels,
      output_device=self.output_device,
    )
    if self.latent_dim == 1:
      return out
    else:
      return out.reshape(samples.shape[0], self.latent_channels, extra_channel_size, -1)

  def encode_tiled_3d(
    self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64)
  ):
    encode_fn = lambda a: self.first_stage_model.encode(
      (self.process_input(a)).to(self.vae_dtype).to(self.device)
    ).float()
    return comfy.utils.tiled_scale_multidim(
      samples,
      encode_fn,
      tile=(tile_t, tile_x, tile_y),
      overlap=overlap,
      upscale_amount=self.downscale_ratio,
      out_channels=self.latent_channels,
      downscale=True,
      index_formulas=self.downscale_index_formula,
      output_device=self.output_device,
    )

  def decode(self, samples_in, vae_options={}):
    self.throw_exception_if_invalid()
    pixel_samples = None
    do_tile = False
    try:
      memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
      model_management.load_models_gpu(
        [self.patcher],
        memory_required=memory_used,
        force_full_load=self.disable_offload,
      )
      free_memory = model_management.get_free_memory(self.device)
      batch_number = int(free_memory / memory_used)
      batch_number = max(1, batch_number)

      for x in range(0, samples_in.shape[0], batch_number):
        samples = samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
        out = self.process_output(
          self.first_stage_model.decode(samples, **vae_options)
          .to(self.output_device)
          .float()
        )
        if pixel_samples is None:
          pixel_samples = torch.empty(
            (samples_in.shape[0],) + tuple(out.shape[1:]), device=self.output_device
          )
        pixel_samples[x : x + batch_number] = out
    except model_management.OOM_EXCEPTION:
      logging.warning(
        "Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding."
      )
      # NOTE: We don't know what tensors were allocated to stack variables at the time of the
      # exception and the exception itself refs them all until we get out of this except block.
      # So we just set a flag for tiler fallback so that tensor gc can happen once the
      # exception is fully off the books.
      do_tile = True

    if do_tile:
      dims = samples_in.ndim - 2
      if dims == 1 or self.extra_1d_channel is not None:
        pixel_samples = self.decode_tiled_1d(samples_in)
      elif dims == 2:
        pixel_samples = self.decode_tiled_(samples_in)
      elif dims == 3:
        tile = 256 // self.spacial_compression_decode()
        overlap = tile // 4
        pixel_samples = self.decode_tiled_3d(
          samples_in, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap)
        )

    pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
    return pixel_samples

  def decode_tiled(
    self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None
  ):
    self.throw_exception_if_invalid()
    memory_used = self.memory_used_decode(
      samples.shape, self.vae_dtype
    )  # TODO: calculate mem required for tile
    model_management.load_models_gpu(
      [self.patcher], memory_required=memory_used, force_full_load=self.disable_offload
    )
    dims = samples.ndim - 2
    args = {}
    if tile_x is not None:
      args["tile_x"] = tile_x
    if tile_y is not None:
      args["tile_y"] = tile_y
    if overlap is not None:
      args["overlap"] = overlap

    if dims == 1:
      args.pop("tile_y")
      output = self.decode_tiled_1d(samples, **args)
    elif dims == 2:
      output = self.decode_tiled_(samples, **args)
    elif dims == 3:
      if overlap_t is None:
        args["overlap"] = (1, overlap, overlap)
      else:
        args["overlap"] = (max(1, overlap_t), overlap, overlap)
      if tile_t is not None:
        args["tile_t"] = max(2, tile_t)

      output = self.decode_tiled_3d(samples, **args)
    return output.movedim(1, -1)

  def encode(self, pixel_samples):
    self.throw_exception_if_invalid()
    pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
    pixel_samples = pixel_samples.movedim(-1, 1)
    do_tile = False
    if self.latent_dim == 3 and pixel_samples.ndim < 5:
      if not self.not_video:
        pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
      else:
        pixel_samples = pixel_samples.unsqueeze(2)
    try:
      memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
      model_management.load_models_gpu(
        [self.patcher],
        memory_required=memory_used,
        force_full_load=self.disable_offload,
      )
      free_memory = model_management.get_free_memory(self.device)
      batch_number = int(free_memory / max(1, memory_used))
      batch_number = max(1, batch_number)
      samples = None
      for x in range(0, pixel_samples.shape[0], batch_number):
        pixels_in = (
          self.process_input(pixel_samples[x : x + batch_number])
          .to(self.vae_dtype)
          .to(self.device)
        )
        out = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
        if samples is None:
          samples = torch.empty(
            (pixel_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device
          )
        samples[x : x + batch_number] = out

    except model_management.OOM_EXCEPTION:
      logging.warning(
        "Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding."
      )
      # NOTE: We don't know what tensors were allocated to stack variables at the time of the
      # exception and the exception itself refs them all until we get out of this except block.
      # So we just set a flag for tiler fallback so that tensor gc can happen once the
      # exception is fully off the books.
      do_tile = True

    if do_tile:
      if self.latent_dim == 3:
        tile = 256
        overlap = tile // 4
        samples = self.encode_tiled_3d(
          pixel_samples, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap)
        )
      elif self.latent_dim == 1 or self.extra_1d_channel is not None:
        samples = self.encode_tiled_1d(pixel_samples)
      else:
        samples = self.encode_tiled_(pixel_samples)

    return samples

  def encode_tiled(
    self,
    pixel_samples,
    tile_x=None,
    tile_y=None,
    overlap=None,
    tile_t=None,
    overlap_t=None,
  ):
    self.throw_exception_if_invalid()
    pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
    dims = self.latent_dim
    pixel_samples = pixel_samples.movedim(-1, 1)
    if dims == 3:
      if not self.not_video:
        pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
      else:
        pixel_samples = pixel_samples.unsqueeze(2)

    memory_used = self.memory_used_encode(
      pixel_samples.shape, self.vae_dtype
    )  # TODO: calculate mem required for tile
    model_management.load_models_gpu(
      [self.patcher], memory_required=memory_used, force_full_load=self.disable_offload
    )

    args = {}
    if tile_x is not None:
      args["tile_x"] = tile_x
    if tile_y is not None:
      args["tile_y"] = tile_y
    if overlap is not None:
      args["overlap"] = overlap

    if dims == 1:
      args.pop("tile_y")
      samples = self.encode_tiled_1d(pixel_samples, **args)
    elif dims == 2:
      samples = self.encode_tiled_(pixel_samples, **args)
    elif dims == 3:
      if tile_t is not None:
        tile_t_latent = max(2, self.downscale_ratio[0](tile_t))
      else:
        tile_t_latent = 9999
      args["tile_t"] = self.upscale_ratio[0](tile_t_latent)

      if overlap_t is None:
        args["overlap"] = (1, overlap, overlap)
      else:
        args["overlap"] = (
          self.upscale_ratio[0](
            max(1, min(tile_t_latent // 2, self.downscale_ratio[0](overlap_t)))
          ),
          overlap,
          overlap,
        )
      maximum = pixel_samples.shape[2]
      maximum = self.upscale_ratio[0](self.downscale_ratio[0](maximum))

      samples = self.encode_tiled_3d(pixel_samples[:, :, :maximum], **args)

    return samples

  def get_sd(self):
    return self.first_stage_model.state_dict()

  def spacial_compression_decode(self):
    try:
      return self.upscale_ratio[-1]
    except:
      return self.upscale_ratio

  def spacial_compression_encode(self):
    try:
      return self.downscale_ratio[-1]
    except:
      return self.downscale_ratio

  def temporal_compression_decode(self):
    try:
      return round(self.upscale_ratio[0](8192) / 8192)
    except:
      return None
