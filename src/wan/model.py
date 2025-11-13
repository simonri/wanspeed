import torch
import torch.nn as nn
import math
from src.wan.attention_block import WanAttentionBlock
from src.flux.layers import EmbedND
from src.wan.head import Head
from src.model_management import cast_to
from src.patcher_extension import get_all_wrappers, WrappersMP, WrapperExecutor
from src.common_dit import pad_to_patch_size


def sinusoidal_embedding_1d(dim, position):
  # preprocess
  assert dim % 2 == 0
  half = dim // 2
  position = position.type(torch.float32)

  # calculation
  sinusoid = torch.outer(
    position, torch.pow(10000, -torch.arange(half).to(position).div(half))
  )
  x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
  return x


class MLPProj(torch.nn.Module):
  def __init__(
    self, in_dim, out_dim, flf_pos_embed_token_number=None, operation_settings={}
  ):
    super().__init__()

    self.proj = torch.nn.Sequential(
      operation_settings.get("operations").LayerNorm(
        in_dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
      operation_settings.get("operations").Linear(
        in_dim,
        in_dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
      torch.nn.GELU(),
      operation_settings.get("operations").Linear(
        in_dim,
        out_dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
      operation_settings.get("operations").LayerNorm(
        out_dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
    )

    if flf_pos_embed_token_number is not None:
      self.emb_pos = nn.Parameter(
        torch.empty(
          (1, flf_pos_embed_token_number, in_dim),
          device=operation_settings.get("device"),
          dtype=operation_settings.get("dtype"),
        )
      )
    else:
      self.emb_pos = None

  def forward(self, image_embeds):
    if self.emb_pos is not None:
      image_embeds = image_embeds[:, : self.emb_pos.shape[1]] + cast_to(
        self.emb_pos[:, : image_embeds.shape[1]],
        dtype=image_embeds.dtype,
        device=image_embeds.device,
      )

    clip_extra_context_tokens = self.proj(image_embeds)
    return clip_extra_context_tokens


class WanModel(torch.nn.Module):
  r"""
  Wan diffusion backbone supporting both text-to-video and image-to-video.
  """

  def __init__(
    self,
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=16,
    num_layers=32,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    flf_pos_embed_token_number=None,
    in_dim_ref_conv=None,
    wan_attn_block_class=WanAttentionBlock,
    image_model=None,
    device=None,
    dtype=None,
    operations=None,
  ):
    r"""
    Initialize the diffusion model backbone.

    Args:
        model_type (`str`, *optional*, defaults to 't2v'):
            Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
        patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
        text_len (`int`, *optional*, defaults to 512):
            Fixed length for text embeddings
        in_dim (`int`, *optional*, defaults to 16):
            Input video channels (C_in)
        dim (`int`, *optional*, defaults to 2048):
            Hidden dimension of the transformer
        ffn_dim (`int`, *optional*, defaults to 8192):
            Intermediate dimension in feed-forward network
        freq_dim (`int`, *optional*, defaults to 256):
            Dimension for sinusoidal time embeddings
        text_dim (`int`, *optional*, defaults to 4096):
            Input dimension for text embeddings
        out_dim (`int`, *optional*, defaults to 16):
            Output video channels (C_out)
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads
        num_layers (`int`, *optional*, defaults to 32):
            Number of transformer blocks
        window_size (`tuple`, *optional*, defaults to (-1, -1)):
            Window size for local attention (-1 indicates global attention)
        qk_norm (`bool`, *optional*, defaults to True):
            Enable query/key normalization
        cross_attn_norm (`bool`, *optional*, defaults to False):
            Enable cross-attention normalization
        eps (`float`, *optional*, defaults to 1e-6):
            Epsilon value for normalization layers
    """

    super().__init__()
    self.dtype = dtype
    operation_settings = {
      "operations": operations,
      "device": device,
      "dtype": dtype,
    }

    assert model_type in ["t2v", "i2v"]
    self.model_type = model_type

    self.patch_size = patch_size
    self.text_len = text_len
    self.in_dim = in_dim
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.freq_dim = freq_dim
    self.text_dim = text_dim
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.cross_attn_norm = cross_attn_norm
    self.eps = eps

    # embeddings
    self.patch_embedding = operations.Conv3d(
      in_dim,
      dim,
      kernel_size=patch_size,
      stride=patch_size,
      device=operation_settings.get("device"),
      dtype=torch.float32,
    )
    self.text_embedding = nn.Sequential(
      operations.Linear(
        text_dim,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
      nn.GELU(approximate="tanh"),
      operations.Linear(
        dim,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
    )

    self.time_embedding = nn.Sequential(
      operations.Linear(
        freq_dim,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
      nn.SiLU(),
      operations.Linear(
        dim,
        dim,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
    )
    self.time_projection = nn.Sequential(
      nn.SiLU(),
      operations.Linear(
        dim,
        dim * 6,
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      ),
    )

    # blocks
    cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
    self.blocks = nn.ModuleList(
      [
        wan_attn_block_class(
          cross_attn_type,
          dim,
          ffn_dim,
          num_heads,
          window_size,
          qk_norm,
          cross_attn_norm,
          eps,
          operation_settings=operation_settings,
        )
        for _ in range(num_layers)
      ]
    )

    # head
    self.head = Head(
      dim, out_dim, patch_size, eps, operation_settings=operation_settings
    )

    d = dim // num_heads
    self.rope_embedder = EmbedND(
      dim=d,
      theta=10000.0,
      axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
    )

    if model_type == "i2v":
      self.img_emb = MLPProj(
        1280,
        dim,
        flf_pos_embed_token_number=flf_pos_embed_token_number,
        operation_settings=operation_settings,
      )
    else:
      self.img_emb = None

    if in_dim_ref_conv is not None:
      self.ref_conv = operations.Conv2d(
        in_dim_ref_conv,
        dim,
        kernel_size=patch_size[1:],
        stride=patch_size[1:],
        device=operation_settings.get("device"),
        dtype=operation_settings.get("dtype"),
      )
    else:
      self.ref_conv = None

  def forward_orig(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
  ):
    r"""
    Forward pass through the diffusion model

    Args:
        x (Tensor):
            List of input video tensors with shape [B, C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [B, L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
      sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype)
    )
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    full_ref = None
    if self.ref_conv is not None:
      full_ref = kwargs.get("reference_latent", None)
      if full_ref is not None:
        full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
        x = torch.concat((full_ref, x), dim=1)

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
      if self.img_emb is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)
      context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.blocks):
      if ("double_block", i) in blocks_replace:

        def block_wrap(args):
          out = {}
          out["img"] = block(
            args["img"],
            context=args["txt"],
            e=args["vec"],
            freqs=args["pe"],
            context_img_len=context_img_len,
            transformer_options=args["transformer_options"],
          )
          return out

        out = blocks_replace[("double_block", i)](
          {
            "img": x,
            "txt": context,
            "vec": e0,
            "pe": freqs,
            "transformer_options": transformer_options,
          },
          {"original_block": block_wrap},
        )
        x = out["img"]
      else:
        x = block(
          x,
          e=e0,
          freqs=freqs,
          context=context,
          context_img_len=context_img_len,
          transformer_options=transformer_options,
        )

    # head
    x = self.head(x, e)

    if full_ref is not None:
      x = x[:, full_ref.shape[1] :]

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x

  def rope_encode(
    self,
    t,
    h,
    w,
    t_start=0,
    steps_t=None,
    steps_h=None,
    steps_w=None,
    device=None,
    dtype=None,
    transformer_options={},
  ):
    patch_size = self.patch_size
    t_len = (t + (patch_size[0] // 2)) // patch_size[0]
    h_len = (h + (patch_size[1] // 2)) // patch_size[1]
    w_len = (w + (patch_size[2] // 2)) // patch_size[2]

    if steps_t is None:
      steps_t = t_len
    if steps_h is None:
      steps_h = h_len
    if steps_w is None:
      steps_w = w_len

    h_start = 0
    w_start = 0
    rope_options = transformer_options.get("rope_options", None)
    if rope_options is not None:
      t_len = (t_len - 1.0) * rope_options.get("scale_t", 1.0) + 1.0
      h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
      w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

      t_start += rope_options.get("shift_t", 0.0)
      h_start += rope_options.get("shift_y", 0.0)
      w_start += rope_options.get("shift_x", 0.0)

    img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
    img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(
      t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype
    ).reshape(-1, 1, 1)
    img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(
      h_start, h_start + (h_len - 1), steps=steps_h, device=device, dtype=dtype
    ).reshape(1, -1, 1)
    img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(
      w_start, w_start + (w_len - 1), steps=steps_w, device=device, dtype=dtype
    ).reshape(1, 1, -1)
    img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

    freqs = self.rope_embedder(img_ids).movedim(1, 2)
    return freqs

  def forward(
    self,
    x,
    timestep,
    context,
    clip_fea=None,
    time_dim_concat=None,
    transformer_options={},
    **kwargs,
  ):
    return WrapperExecutor.new_class_executor(
      self._forward,
      self,
      get_all_wrappers(WrappersMP.DIFFUSION_MODEL, transformer_options),
    ).execute(
      x,
      timestep,
      context,
      clip_fea,
      time_dim_concat,
      transformer_options,
      **kwargs,
    )

  def _forward(
    self,
    x,
    timestep,
    context,
    clip_fea=None,
    time_dim_concat=None,
    transformer_options={},
    **kwargs,
  ):
    bs, c, t, h, w = x.shape
    x = pad_to_patch_size(x, self.patch_size)

    t_len = t
    if time_dim_concat is not None:
      time_dim_concat = pad_to_patch_size(time_dim_concat, self.patch_size)
      x = torch.cat([x, time_dim_concat], dim=2)
      t_len = x.shape[2]

    if self.ref_conv is not None and "reference_latent" in kwargs:
      t_len += 1

    freqs = self.rope_encode(
      t_len,
      h,
      w,
      device=x.device,
      dtype=x.dtype,
      transformer_options=transformer_options,
    )
    return self.forward_orig(
      x,
      timestep,
      context,
      clip_fea=clip_fea,
      freqs=freqs,
      transformer_options=transformer_options,
      **kwargs,
    )[:, :, :t, :h, :w]

  def unpatchify(self, x, grid_sizes):
    r"""
    Reconstruct video tensors from patch embeddings.

    Args:
        x (List[Tensor]):
            List of patchified features, each with shape [L, C_out * prod(patch_size)]
        grid_sizes (Tensor):
            Original spatial-temporal grid dimensions before patching,
                shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

    Returns:
        List[Tensor]:
            Reconstructed video tensors with shape [L, C_out, F, H / 8, W / 8]
    """

    c = self.out_dim
    u = x
    b = u.shape[0]
    u = u[:, : math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
    u = torch.einsum("bfhwpqrc->bcfphqwr", u)
    u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
    return u
