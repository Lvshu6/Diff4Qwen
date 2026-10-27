"""
LayerEditDiT: image -> Live2D layered PSD (segmentation + generation).

The architecture follows the Wan-Video DiT. Video generation is migrated to
multi-layer generation by repurposing the temporal axis as the layer order:

    latents: (B, C, 1 + n, H, W)
        frame 0      -> layer_input_image (clean condition, timestep = 0)
        frame 1..n   -> the n segmented layers to be generated (noisy)

Just like image-to-video generation (Wan2.2-TI2V), the condition image is fused
into the first frame of the latents and the pending-to-generate noise is
appended after it. Each token carries its own timestep, so the condition frame
is always treated as clean. The 3D RoPE assigns the layer index to the
temporal dimension, which lets the model reason about the layer order.

Input tokens (fused-overwrite, the condition layer is always clean):
    latents  : latents (B, 16, 1 + n, h, w) ++ masks (B, 4, 1 + n, h, w) on channels
               -> patchify -> (B, (1 + n) * h/2 * w/2, 20 * 4)
    condition: first_frame_latents (layer_input_image, B, 16, 1, h, w) ++ all-ones mask (B, 4, 1, h, w)
               -> patchify -> (B, h/2 * w/2, 20 * 4), repeated for each of the 1 + n frames
    tokens   : latents ++ condition on channels -> (B, (1 + n) * h/2 * w/2, 2 * 20 * 4)
               -> dit.img_in (Linear) -> (B, L, dim)
`patchify` is parameter-free: every 2x2 patch is flattened into the channel dimension.
`masks` (B, 4, 1 + n, h, w) are aligned with the 1 + n frames: single channel in [0, 1] (white = 1),
downsampled to the latent resolution and replicated to 4 channels (see LayerEditUnit_MaskEmbedder).
Missing masks, including the default mask of layer_input_image, are all ones.

The 30 DiT blocks follow a cyclic pattern A, RGB, RGB repeated 10 times:
    A   (blocks 0, 3, ..., 27): self-attention + text-cross-attention
    RGB (all other blocks)    : self-attention only

Latents come from the RGBA VAE of Qwen-Image-Layered (z_dim = 16) and the text
context comes from the Qwen-Image text encoder (hidden size = 3584).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from ..core.gradient import gradient_checkpoint_forward

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


LAYER_EDIT_DIT_PRESETS = {
    # All presets: 30 layers = 10 cycles of (A, RGB, RGB).
    # Same widths as Wan2.1-T2V-1.3B
    "small": {"dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30, "cycle_length": 3},
    # Same widths as Wan2.2-TI2V-5B
    "base": {"dim": 3072, "ffn_dim": 14336, "num_heads": 24, "num_layers": 30, "cycle_length": 3},
    # Same widths as Wan2.1-T2V-14B
    "large": {"dim": 5120, "ffn_dim": 13824, "num_heads": 40, "num_layers": 30, "cycle_length": 3},
}


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute: (layer, height, width)
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, has_text_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.has_text_attn = has_text_attn

        self.self_attn = SelfAttention(dim, num_heads, eps)
        if has_text_attn:
            self.cross_attn = CrossAttention(dim, num_heads, eps)
            self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        # t_mod: (B, L, 6, dim), one modulation vector per token
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        if self.has_text_attn:
            x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t):
        # t: (B, L, dim)
        shift, scale = (self.modulation.unsqueeze(0).to(dtype=t.dtype, device=t.device) + t.unsqueeze(2)).chunk(2, dim=2)
        x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        return x


class LayerEditDiT(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int = 1536,
        in_dim: int = 16,
        mask_dim: int = 4,
        ffn_dim: int = 8960,
        out_dim: int = 16,
        text_dim: int = 3584,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 12,
        num_layers: int = 30,
        cycle_length: int = 3,
        rope_max_len: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.mask_dim = mask_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        # A blocks: self + text attention. RGB blocks: self-attention only.
        # The first block of every `cycle_length` consecutive blocks is an A block: A, RGB, RGB, A, ...
        self.cycle_length = cycle_length

        # Parameter-free patchify (see `patchify`), then img_in maps the flattened patches of
        # [latents ++ masks] and [first_frame_latents ++ all-ones mask] (concatenated on channels) to `dim`.
        self.img_in = nn.Linear(2 * (in_dim + mask_dim) * math.prod(self.patch_size), dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps, has_text_attn=layer_id % cycle_length == 0)
            for layer_id in range(num_layers)
        ])
        self.head = Head(dim, out_dim, self.patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim, end=rope_max_len)

    @torch.no_grad()
    def init_weights(self):
        # Same initialization strategy as the official Wan-Video DiT.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.head.head.bias)

    def patchify(self, x: torch.Tensor):
        # Parameter-free: (B, C, F, H, W) -> (B, F * H/2 * W/2, 1 * 2 * 2 * C), the inverse of `unpatchify`
        p, q, r = self.patch_size
        f, h, w = x.shape[2] // p, x.shape[3] // q, x.shape[4] // r
        x = rearrange(x, 'b c (f p) (h q) (w r) -> b (f h w) (p q r c)', p=p, q=q, r=r).contiguous()
        return x, (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def build_freqs(self, grid_size: Tuple[int, int, int], device, frame_offset: int = 0):
        f, h, w = grid_size
        freqs = torch.cat([
            self.freqs[0][frame_offset:frame_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)
        return freqs

    def build_masks(self, masks: Optional[torch.Tensor], x: torch.Tensor):
        # masks are aligned with the frames of x; without masks, every frame gets an all-ones mask.
        b, _, f, h, w = x.shape
        if masks is None:
            return torch.ones((b, self.mask_dim, f, h, w), dtype=x.dtype, device=x.device)
        if masks.shape[2] != f:
            raise ValueError(f"Got {masks.shape[2]} masks, but x has {f} frames (layer_input_image + layers).")
        if masks.shape[0] != b:
            masks = masks.repeat(b // masks.shape[0], 1, 1, 1, 1)
        return masks.to(dtype=x.dtype, device=x.device)

    def prepare_block_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_condition_layers: int = 1,
    ):
        # x: (B, C, num_condition_layers + n, H, W), frame 0 is the clean first_frame_latents (overwritten).
        # Masks are concatenated on channels before patchify.
        batch_size = x.shape[0]
        if num_condition_layers > 1:
            raise ValueError(f"Only one condition layer (layer_input_image) is supported, got {num_condition_layers}.")
        # layer_input_image is encoded on its own as an extra condition (like Qwen-Image-Layered).
        # first_frame_latents ++ all-ones mask -> patchify, then concatenated to the tokens of every frame on channels.
        condition = x[:, :, :1] if num_condition_layers > 0 else torch.zeros_like(x[:, :, :1])
        condition = torch.cat([condition, torch.ones_like(condition[:, :1]).repeat(1, self.mask_dim, 1, 1, 1)], dim=1)
        condition_tokens, _ = self.patchify(condition)
        x = torch.cat([x, self.build_masks(masks, x)], dim=1)
        x, grid = self.patchify(x)
        # Tokens are ordered frame by frame, so repeating along the sequence aligns the condition with every frame.
        x = self.img_in(torch.cat([x, condition_tokens.repeat(1, grid[0], 1)], dim=2))
        freqs = self.build_freqs(grid, x.device)
        timestep = timestep.reshape(-1)
        if timestep.shape[0] == 1:
            timestep = timestep.repeat(batch_size)
        timestep = timestep.view(batch_size, 1).repeat(1, grid[0]).clone()
        # The condition layer is clean, so its timestep is 0.
        timestep[:, :num_condition_layers] = 0
        timestep = timestep.unsqueeze(2).repeat(1, 1, grid[1] * grid[2]).flatten(1)

        if context.shape[0] != batch_size:
            context = context.repeat(batch_size // context.shape[0], 1, 1)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(x.dtype)
        ).unflatten(0, (batch_size, -1))
        t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
        context = self.text_embedding(context)

        block_kwargs = {"context": context, "t_mod": t_mod, "freqs": freqs}
        return x, t, grid, block_kwargs

    def forward_blocks(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        hidden_states: Optional[list] = None,
    ):
        # Run blocks[start:end]. If `hidden_states` is a list, the output of every block is appended to it.
        for block in self.blocks[start:end]:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing and self.training,
                use_gradient_checkpointing_offload and self.training,
                x, context, t_mod, freqs
            )
            if hidden_states is not None:
                hidden_states.append(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_condition_layers: int = 1,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_hidden_states: bool = False,
        **kwargs,
    ):
        """
        x: (B, in_dim, 1 + n, H, W). The first `num_condition_layers` frames are the clean
            layer_input_image latents (overwritten), the rest are the noisy latents of the layers to be generated.
        timestep: (B,) or (1,), the timestep of the noisy layers.
        context: (B, S, text_dim), text embeddings from the Qwen-Image text encoder.
        masks: (B, mask_dim, 1 + n, H, W) in [0, 1], optional. Segmentation masks at the latent resolution,
            aligned with the frames of x. Defaults to all ones.
        return_hidden_states: if True, also return `hidden_states` (hidden_states[i] is the input of
            block i, hidden_states[-1] is the output of the last block) and `block_kwargs`. They are
            only used to route gradients to specific blocks during training (see LayerEditRGBALoss).
        Returns (B, out_dim, 1 + n, H, W), in the same frame order as `x`.
        """
        x, t, grid, block_kwargs = self.prepare_block_inputs(x, timestep, context, masks, num_condition_layers)
        hidden_states = [x] if return_hidden_states else None
        x = self.forward_blocks(
            x, 0, len(self.blocks), **block_kwargs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            hidden_states=hidden_states,
        )
        x = self.head(x, t)
        x = self.unpatchify(x, grid)
        if return_hidden_states:
            return x, hidden_states, block_kwargs
        return x
