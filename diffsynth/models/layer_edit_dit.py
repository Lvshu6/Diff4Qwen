"""
LayerEditDiT: image -> Live2D layered PSD (segmentation + generation).

The DiT replicates Qwen-Image-Layered (QwenImageDiT with use_layer3d_rope and use_additional_t_cond, run by
model_fn_qwen_image) as closely as possible, so that its weights are an effective initialization
(examples/layer_edit/model_training/dit_init.py). The differences are the mask input, the sparse tokens and the
text stream (see below).

    latents  : (B, 16, 1 + n, H, W), all frames noisy, all generated, as in Qwen-Image-Layered:
                   frame 0      -> the composite image (layer_input_image)
                   frame 1..n   -> the n layers
    condition: condition_latents (B, 16, 1, H, W), the clean layer_input_image, appended after the latent tokens
               (layer_input_latents of Qwen-Image-Layered).

Like Qwen-Image-Layered:
    - Layer-order encoding: QwenEmbedLayer3DRope, axes (16, 56, 56) per 128-dim head, theta 10000, height /
      width centred (scale_rope). Latent frame i takes the layer index i, the condition image the index -1.
      The text tokens take the positions after max(h / 2, w / 2, number of latent frames), on all three axes.
    - Timestep: every token, the condition included, is modulated with the noisy timestep (zero_cond_t=False,
      the default of Qwen-Image-Layered). `zero_cond_t=True` gives the condition tokens timestep 0 instead.
    - Timestep embedding: sinusoidal(256, cos first, t in [0, 1000]) -> Linear -> SiLU -> Linear, plus the
      constant addition_t_embedding[0] (folded into the last bias by dit_init.py). Every block has its own img_mod.
    - Blocks: the image stream of QwenImageTransformerBlock with the same names, shapes and eps; head = norm_out
      (AdaLayerNorm, single) + proj_out. Patch features are (P Q C) here and (C P Q) in Qwen, dit_init.py permutes.

Differences:
    - Mask: a single channel (B, 1, 1 + n, h, w), concatenated to the latents on channels before patchify; the
      condition gets an all-ones mask. Its img_in columns are zero-initialized, so at initialization the mask
      changes nothing.
    - Sparse tokens (`sparse_tokens=True`, the default): only the latent tokens whose 2x2 patch has a non-zero mask
      value enter img_in and the blocks, with the RoPE positions of their original places. The head output is
      spliced back to the full (1 + n) * h/2 * w/2 tokens with zeros at the dropped tokens (the pipeline fills them).
    - Text stream: the text tokens (context -> txt_norm + txt_in) are not updated by the blocks. In every block
      they are normalized and modulated like in Qwen (txt_norm1, txt_mod: shift / scale with the noisy timestep),
      then only give keys / values (add_k_proj, norm_added_k + RoPE, add_v_proj) to the image tokens, which attend
      to [text, image] in one softmax. Dropped: add_q_proj, norm_added_q, to_add_out, txt_norm2, txt_mlp, the
      text gates and the MLP rows of txt_mod. The first block is therefore exact; later blocks differ from Qwen only
      because Qwen's text has been updated by the blocks before.

`patchify` is parameter-free: every 2x2 patch is flattened into the channel dimension. Blocks are grouped in the
cycle A, B, B only to route the RGB / alpha losses in training (`cycle_length`), the blocks are all the same.
Latents come from the RGBA VAE of Qwen-Image-Layered (z_dim = 16), the text from the Qwen-Image text encoder (3584).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from ..core.gradient import gradient_checkpoint_forward
from .qwen_image_dit import QwenEmbedLayer3DRope

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


def rope_apply(x, freqs, num_heads):
    # Same as apply_rotary_emb_qwen: adjacent pairs, float32. x: (B, S, num_heads * head_dim), freqs: (S, head_dim / 2) complex.
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    x_out = torch.view_as_real(x_out * freqs.unsqueeze(1)).flatten(2)
    return x_out.type_as(x)


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


def select_timestep(x: torch.Tensor, clean: torch.Tensor):
    # x: (B, 2, ...) computed for [noisy timestep, 0], clean: (L,) bool -> (B, L, ...), clean tokens take index 1.
    return torch.where(clean.view(1, -1, *([1] * (x.dim() - 2))), x[:, 1:2], x[:, 0:1])


class ApproximateGELU(nn.Module):
    # Same as Qwen-Image: x * sigmoid(1.702 x).
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    # Same layout as QwenFeedForward (net.0.proj, net.2).
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.net = nn.ModuleList([ApproximateGELU(dim, ffn_dim), nn.Dropout(0.0), nn.Linear(ffn_dim, dim)])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class JointAttention(nn.Module):
    """
    Qwen-Image joint attention without the text stream: only the image tokens are queries, the keys / values
    are [text, image] in one softmax. The text tokens are projected to keys / values but never updated, so
    add_q_proj, norm_added_q and to_add_out of Qwen are dropped. The text keys get the text RoPE of Qwen.
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.norm_added_k = RMSNorm(self.head_dim, eps=eps)

        self.to_out = nn.Sequential(nn.Linear(dim, dim))

        self.attn = AttentionModule(self.num_heads)

    def head_norm(self, norm: RMSNorm, x: torch.Tensor):
        # Per-head RMSNorm, like Qwen-Image.
        return norm(rearrange(x, "b s (n d) -> b s n d", n=self.num_heads)).flatten(2)

    def forward(self, x: torch.Tensor, context: torch.Tensor, freqs: torch.Tensor, txt_freqs: torch.Tensor):
        q = self.head_norm(self.norm_q, self.to_q(x))
        k = self.head_norm(self.norm_k, self.to_k(x))
        v = self.to_v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        txt_k = rope_apply(self.head_norm(self.norm_added_k, self.add_k_proj(context)), txt_freqs, self.num_heads)
        k = torch.cat([txt_k, k], dim=1)
        v = torch.cat([self.add_v_proj(context), v], dim=1)
        x = self.attn(q, k, v)
        return self.to_out(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    # The image stream of QwenImageTransformerBlock, with the same parameter names. The text tokens are never
    # updated (txt_norm2, txt_mlp, the text gates are removed), but before they give keys / values they are
    # normalized and modulated like in Qwen: LN(text) * (1 + scale) + shift, with txt_norm1 and txt_mod.
    # txt_mod keeps only the (shift, scale) rows of Qwen's txt_mod for the attention (2 * dim of its 6 * dim).
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Own timestep projection in every block, like Qwen-Image.
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.txt_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = JointAttention(dim, num_heads, eps)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.img_mlp = FeedForward(dim, ffn_dim)
        self.gate = GateModule()

    def forward(self, x, context, temb, clean, freqs, txt_freqs):
        # temb: (B, 2, dim), the embeddings of [noisy timestep, 0]. clean: (L,) bool, the tokens with timestep 0.
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = select_timestep(self.img_mod(temb), clean).chunk(6, dim=-1)
        input_x = modulate(self.img_norm1(x), shift_msa, scale_msa)
        # The text is modulated with the noisy timestep, as Qwen's txt_mod (which never takes the zero_cond_t index)
        txt_shift, txt_scale = self.txt_mod(temb[:, 0]).unsqueeze(1).chunk(2, dim=-1)
        input_context = modulate(self.txt_norm1(context), txt_shift, txt_scale)
        x = self.gate(x, gate_msa, self.attn(input_x, input_context, freqs, txt_freqs))
        input_x = modulate(self.img_norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.img_mlp(input_x))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # Same as Qwen-Image norm_out (AdaLayerNorm, single): linear(silu(temb)) -> (scale, shift).
        self.linear = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))

    def forward(self, x, temb, clean):
        scale, shift = select_timestep(self.linear(F.silu(temb)), clean).chunk(2, dim=-1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class LayerEditDiT(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int = 3072,
        in_dim: int = 16,
        mask_dim: int = 1,
        ffn_dim: int = 12288,
        out_dim: int = 16,
        text_dim: int = 3584,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 24,
        num_layers: int = 60,
        cycle_length: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.mask_dim = mask_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        # Every block attends to the text. `cycle_length` only groups the blocks for the RGB / alpha losses
        # in training: the first block of every `cycle_length` consecutive blocks is an A block: A, B, B, A, ...
        self.cycle_length = cycle_length

        # Parameter-free patchify (see `patchify`), then img_in maps the flattened patches of
        # [latents ++ masks] and [first_frame_latents ++ all-ones mask] (concatenated on the sequence) to `dim`.
        self.img_in = nn.Linear((in_dim + mask_dim) * math.prod(self.patch_size), dim)
        # Qwen-Image txt_norm + txt_in. The text tokens are not updated after this.
        self.text_embedding = nn.Sequential(
            RMSNorm(text_dim, eps=eps),
            nn.Linear(text_dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, self.patch_size, eps)
        # The layer-order RoPE of Qwen-Image-Layered: axes (16, 56, 56) of the 128-dim heads.
        self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.text_embedding[1].weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.head.head.bias)
        # The mask columns of img_in start at zero: at initialization the masks do not change the copied embedding.
        self.img_in.weight.view(self.dim, -1, self.in_dim + self.mask_dim)[:, :, self.in_dim:].zero_()

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

    def build_freqs(self, num_frames: int, height: int, width: int, with_condition: bool, text_len: int, device):
        """The RoPE of Qwen-Image-Layered (QwenEmbedLayer3DRope.forward): latent frame i -> layer index i, the
        condition image -> index -1, the text -> positions from max(h / 2, w / 2, number of latent frames).
        Returns (image freqs (num_frames [+ 1]) * h * w, text freqs text_len), complex (., 64)."""
        rope = self.pos_embed
        if rope.pos_freqs.device != device:
            rope.pos_freqs, rope.neg_freqs = rope.pos_freqs.to(device), rope.neg_freqs.to(device)
        freqs = [rope._compute_video_freqs(1, height, width, idx).to(device) for idx in range(num_frames)]
        if with_condition:
            freqs.append(rope._compute_condition_freqs(1, height, width).to(device))
        # layer_num of Qwen = number of image entries - 1 (= the latent frames when the condition is there)
        max_vid_index = max(height // 2, width // 2, len(freqs) - 1)
        txt_freqs = rope.pos_freqs[max_vid_index: max_vid_index + text_len]
        return torch.cat(freqs, dim=0), txt_freqs

    def build_masks(self, masks: Optional[torch.Tensor], x: torch.Tensor):
        # masks are aligned with the frames of x; without masks, every frame gets an all-ones mask.
        b, _, f, h, w = x.shape
        if masks is None:
            return torch.ones((b, self.mask_dim, f, h, w), dtype=x.dtype, device=x.device)
        if masks.shape[2] != f:
            raise ValueError(f"Got {masks.shape[2]} masks, but x has {f} frames (layer_input_image + layers).")
        if masks.shape[1] != self.mask_dim:
            # A single mask channel: masks cached with the channel replicated (older caches) are averaged back to one.
            masks = masks.float().mean(dim=1, keepdim=True).repeat(1, self.mask_dim, 1, 1, 1)
        if masks.shape[0] != b:
            masks = masks.repeat(b // masks.shape[0], 1, 1, 1, 1)
        return masks.to(dtype=x.dtype, device=x.device)

    def select_tokens(self, masks: torch.Tensor):
        # masks: (B, mask_dim, F, H, W) -> (F * H/2 * W/2,) bool. A token is kept if any cell of its patch is non-zero,
        # so every region with mask = 1 is covered. The selection is shared by the batch (union of the masks).
        return self.patchify(masks)[0].amax(dim=(0, 2)) > 0

    def kept_latents(self, masks: Optional[torch.Tensor], x: torch.Tensor):
        # x: (B, C, F, H, W) -> (1, 1, F, H, W) bool, the latent cells whose tokens enter the DiT.
        masks = self.build_masks(masks, x)
        tokens, grid = self.patchify(masks)
        keep = self.select_tokens(masks).view(1, -1, 1).expand(-1, -1, tokens.shape[2] // masks.shape[1])
        return self.unpatchify(keep, grid)

    def prepare_block_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        sparse_tokens: bool = True,
        zero_cond_t: bool = False,
    ):
        # x: (B, C, 1 + n, H, W), all frames noisy. condition_latents: (B, C, 1, H, W), clean, appended on the sequence.
        batch_size = x.shape[0]
        masks = self.build_masks(masks, x)
        # With sparse_tokens=False, the masks still enter as channels, but every token is kept.
        keep = self.select_tokens(masks if sparse_tokens else torch.ones_like(masks))
        tokens = [torch.cat([x, masks], dim=1)]
        if condition_latents is not None:
            if condition_latents.shape[0] != batch_size:
                condition_latents = condition_latents.repeat(batch_size // condition_latents.shape[0], 1, 1, 1, 1)
            condition_latents = condition_latents.to(dtype=x.dtype, device=x.device)
            tokens.append(torch.cat([condition_latents, torch.ones_like(condition_latents[:, :1]).repeat(1, self.mask_dim, 1, 1, 1)], dim=1))
        tokens = [self.patchify(t) for t in tokens]
        grid = tokens[0][1]
        num_condition_tokens = tokens[1][0].shape[1] if condition_latents is not None else 0
        # Drop the latent tokens outside the masks. The condition tokens are always kept.
        keep_all = torch.cat([keep, torch.ones(num_condition_tokens, dtype=torch.bool, device=keep.device)])
        x = self.img_in(torch.cat([t for t, _ in tokens], dim=1)[:, keep_all])

        if context.shape[0] != batch_size:
            context = context.repeat(batch_size // context.shape[0], 1, 1)
        freqs, txt_freqs = self.build_freqs(grid[0], grid[1], grid[2], condition_latents is not None, context.shape[1], x.device)
        freqs = freqs[keep_all]
        # Every token takes the noisy timestep (Qwen-Image-Layered); with zero_cond_t, the condition tokens take 0.
        clean = torch.zeros(keep_all.shape[0], dtype=torch.bool, device=x.device)
        if zero_cond_t:
            clean[keep.shape[0]:] = True
        clean = clean[keep_all]
        timestep = timestep.reshape(-1)
        if timestep.shape[0] == 1:
            timestep = timestep.repeat(batch_size)
        timestep = torch.stack([timestep, torch.zeros_like(timestep)], dim=1)
        # temb: (B, 2, dim), the embeddings of [noisy timestep, 0]. Every block projects it on its own.
        temb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(x.dtype)
        ).unflatten(0, (batch_size, 2))
        context = self.text_embedding(context)

        block_kwargs = {"context": context, "temb": temb, "clean": clean, "freqs": freqs, "txt_freqs": txt_freqs}
        return x, grid, keep, block_kwargs

    def forward_blocks(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        context: torch.Tensor,
        temb: torch.Tensor,
        clean: torch.Tensor,
        freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
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
                x, context, temb, clean, freqs, txt_freqs
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
        condition_latents: Optional[torch.Tensor] = None,
        sparse_tokens: bool = True,
        zero_cond_t: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_hidden_states: bool = False,
        **kwargs,
    ):
        """
        x: (B, in_dim, 1 + n, H, W), the noisy latents: frame 0 the composite image, frames 1..n the layers.
        timestep: (B,) or (1,), in [0, 1000].
        condition_latents: (B, in_dim, 1, H, W), the clean layer_input_image latents, appended as condition tokens.
        context: (B, S, text_dim), text embeddings from the Qwen-Image text encoder.
        masks: (B, mask_dim, 1 + n, H, W) in [0, 1], optional. Segmentation masks at the latent resolution,
            aligned with the frames of x. Defaults to all ones.
        sparse_tokens: if True, tokens outside the masks do not enter the blocks. If False, all tokens do.
        zero_cond_t: if True, the condition tokens are modulated with timestep 0 (Qwen-Image-Layered: False).
        return_hidden_states: if True, also return `hidden_states` (hidden_states[i] is the input of
            block i, hidden_states[-1] is the output of the last block) and `block_kwargs`. They are
            only used to route gradients to specific blocks during training (see LayerEditRGBALoss).
        Returns (B, out_dim, 1 + n, H, W), in the same frame order as `x`, zero at the dropped tokens.
        """
        x, grid, keep, block_kwargs = self.prepare_block_inputs(x, timestep, context, masks, condition_latents, sparse_tokens, zero_cond_t)
        hidden_states = [x] if return_hidden_states else None
        x = self.forward_blocks(
            x, 0, len(self.blocks), **block_kwargs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            hidden_states=hidden_states,
        )
        # Drop the appended condition tokens, only the kept tokens of the 1 + n latent frames go through the head.
        num_tokens = int(keep.sum())
        x = self.head(x[:, :num_tokens], block_kwargs["temb"], block_kwargs["clean"][:num_tokens])
        # Splice the tokens back to the complete latent.
        output = x.new_zeros((x.shape[0], keep.shape[0], x.shape[2]))
        output[:, keep] = x
        x = self.unpatchify(output, grid)
        if return_hidden_states:
            return x, hidden_states, block_kwargs
        return x
