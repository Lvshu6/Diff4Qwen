# LayerEditDiT specification

Create the model architecture `LayerEditDiT` for image-to-Live2D-layered-PSD segmentation and generation. Input: one illustration (`layer_input_image`). Output: `n` RGBA layers.

## Files

| File | Purpose |
|---|---|
| `diffsynth/models/layer_edit_dit.py` | The `LayerEditDiT` model |
| `diffsynth/pipelines/layer_edit.py` | `LayerEditPipeline`, its units, and the RGB / alpha loss |
| `diffsynth/configs/model_configs.py` | `layer_edit_series` entry: model hash and `extra_kwargs` |
| `diffsynth/configs/vram_management_module_maps.py` | VRAM management map for `LayerEditDiT` |
| `examples/layer_edit/model_training/dit_init.py` | Builds the initial weights from Qwen-Image-Layered |
| `examples/layer_edit/model_training/train.py` | Training entry point |
| `examples/layer_edit/model_training/layer_edit_full.sh` | Caching and training commands, based on `examples/qwen_image/model_training/Qwen-Image-Layered_distill.sh` |
| `examples/layer_edit/model_training/full/accelerate_config_14B.yaml` | Copy of `examples/wanvideo/model_training/full/accelerate_config_14B.yaml` with `num_processes: 1` |
| `examples/layer_edit/model_training/full/accelerate_config_zero3.yaml` | Copy of `examples/wanvideo/model_training/full/accelerate_config_zero3.yaml` with `num_processes: 1` |
| `examples/layer_edit/model_inference_low_vram/LayerEdit.py` | Low-VRAM batch inference |

## Reused models

- **VAE:** `QwenImageVAE` in `diffsynth/models/qwen_image_vae.py`, the RGBA VAE of Qwen-Image-Layered with `image_channels=4` and `z_dim=16`. Load `Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors`. The latent is downsampled 8x.
- **Text encoder:** `QwenImageTextEncoder` in `diffsynth/models/qwen_image_text_encoder.py`, hidden size 3584. Load `Qwen/Qwen-Image:text_encoder/model*.safetensors`.
- **Tokenizer:** `Qwen/Qwen-Image:tokenizer/`.

## DiT architecture

Replicate the Qwen-Image-Layered DiT (`QwenImageDiT` with `use_layer3d_rope=True`, `use_additional_t_cond=True`, as run by `model_fn_qwen_image`) as closely as possible, so its weights are an effective initialization. LayerEdit adds the mask on top of this basis: a single mask channel, and sparse token input based on the mask. The only other difference is that the text tokens are not updated by the blocks. Everything else keeps Qwen's widths, parameter names, activations, layer-order encoding, and timestep handling.

Check (`dit_init.py` mapping, random Qwen weights, `sparse_tokens=False`, any masks since their columns are zero): the first block reproduces Qwen-Image-Layered to float32 rounding with or without text (relative difference 4e-6), and the whole model does without text tokens. With text, the later blocks differ only because Qwen's text has been updated by the blocks before.

Defaults: `dim=3072`, `ffn_dim=12288`, `num_heads=24` (head dim 128), `num_layers=60`, `in_dim=16`, `mask_dim=1`, `out_dim=16`, `text_dim=3584`, `freq_dim=256`, `patch_size=(1, 2, 2)`, `cycle_length=3`, `eps=1e-6`.

**Total: 12,501,143,616 parameters.**

### Text

- `text_embedding` is `RMSNorm(3584)` followed by `Linear(3584, dim)`, which correspond to Qwen's `txt_norm` and `txt_in`.
- No block updates the text tokens after `text_embedding`; the same tokens go to every block and nothing is written back.
- In every block, before giving keys and values, the text is normalized and modulated like in Qwen: `LN(text) * (1 + scale) + shift`, with `txt_norm1` (`LayerNorm` without parameters) and `txt_mod = Sequential(SiLU, Linear(dim, 2 * dim))` on the noisy-timestep `temb`. `txt_mod` holds only the (shift, scale) rows of Qwen's `txt_mod` for the attention (rows 0 to 2·dim of its 6·dim).
- Keep `attn.add_k_proj`, `attn.add_v_proj`, and `attn.norm_added_k` (then the text RoPE). They only derive keys and values for the attention in that block.
- Remove `txt_norm2`, `txt_mlp`, the text gates and MLP rows of `txt_mod`, `attn.add_q_proj`, `attn.norm_added_q`, and `attn.to_add_out`.

### Blocks

All 60 `DiTBlock`s are identical and contain:

- **Timestep projection:** each block has its own `img_mod = Sequential(SiLU, Linear(dim, 6 * dim))`, as in Qwen. Do not share one projection across blocks.
- **Attention:** `JointAttention`. Only image tokens are queries. Keys and values are `[text, image]` concatenated on the sequence and share one softmax. RoPE applies to image queries and keys and, with Qwen's text positions, to text keys. Keep Qwen's per-head `RMSNorm` (`norm_q`, `norm_k`, `norm_added_k`); RoPE is applied after the norm, in float32, as `apply_rotary_emb_qwen`.
- **MLP:** `img_mlp` with Qwen's approximate GELU, `x * sigmoid(1.702 * x)`. Parameter names are `net.0.proj` and `net.2`.
- **Norms:** `img_norm1` and `img_norm2`, `LayerNorm` without learnable parameters.

Other modules:

- `time_embedding`: sinusoidal embedding of `freq_dim` (cos first, t in [0, 1000]), then `Linear`, `SiLU`, `Linear`, the same as Qwen's `time_text_embed`; Qwen's constant `addition_t_embedding[0]` is folded into the last bias.
- `Head`: non-learnable `LayerNorm`, `linear = Linear(dim, 2 * dim)` applied to `silu(temb)` to give (scale, shift), and `head = Linear(dim, 64)` (Qwen's `norm_out`, `proj_out`).
- RoPE: Qwen's `QwenEmbedLayer3DRope` (`self.pos_embed`), axes (16, 56, 56) of the 128-dim head for (layer, height, width), theta 10000, height and width centred (`scale_rope=True`). Latent frame i takes layer index i; the condition image takes layer index -1. Text tokens take positions from `max(h/2, w/2, number of latent frames)` on all three axes. Do not use a Wan-style RoPE: the layer-order encoding must be identical to Qwen-Image-Layered.

### Layer sequence

As in Qwen-Image-Layered, latents have shape `(B, 16, 1 + n, h, w)` and every frame is noisy and generated:

- Frame 0 is the composite image (`layer_input_image`). It is denoised like the layers, never overwritten; its training target is the composite.
- Frames 1 to n are the layers.
- The clean `layer_input_image` latents (`first_frame_latents`, Qwen's `layer_input_latents`) are the condition, passed to the DiT as `condition_latents` and appended as condition tokens.

### Input tokens

`patchify` has no parameters. It reshapes each 2x2 patch into the channel dimension: `(B, C, F, H, W)` to `(B, F * H/2 * W/2, 4 * C)`.

1. **Latent tokens:** concatenate latents `(B, 16, 1 + n, h, w)` and the single-channel mask `(B, 1, 1 + n, h, w)` on channels, then patchify.
2. **Condition tokens:** take `condition_latents` (`first_frame_latents` of `layer_input_image`, `(B, 16, 1, h, w)`), concatenate an all-ones mask `(B, 1, 1, h, w)` on channels, then patchify.
3. **Sequence:** concatenate latent tokens and condition tokens on the token sequence dimension, giving `(B, (1 + n + 1) * h/2 * w/2, 68)`.
4. **Projection:** apply `img_in = Linear(68, dim)`.

The appended condition tokens take RoPE layer index -1, as in Qwen-Image-Layered.

### Sparse tokens

The VAE needs a complete latent, but the DiT only needs tokens, so masks reduce the tokens entering the DiT:

- Keep a latent token if any cell of its 2x2 patch has a mask value above 0. Patches partly covered by a mask are kept, so every region with mask = 1 is covered.
- Select tokens after patchify and before `img_in`. Kept tokens keep the RoPE positions and timesteps of their original places. Appended condition tokens are always kept.
- With no `mask`, or all-white masks, no tokens are dropped.
- In training, `--sparse_tokens` is on by default. `--no-sparse_tokens` keeps all tokens while masks still enter as channels, and needs no re-caching.
- The selection is shared across a batch, as the union of its masks.
- This adds no parameters, so existing weights and the model hash are unaffected.

### Timesteps and output

- Every token, the condition tokens included, uses the noisy timestep, as Qwen-Image-Layered (`zero_cond_t=False`).
- `zero_cond_t=True` gives the appended condition tokens timestep 0 instead: `temb` is computed for `[noisy timestep, 0]` and each block's `img_mod` selects per token.
- Drop the appended condition tokens before `Head`. Splice the kept tokens back into a complete token sequence, with zeros at dropped tokens, then unpatchify. The output is `(B, 16, 1 + n, h, w)`, in the same frame order as the input latents.

## Pipeline units

`LayerEditPipeline` uses `FlowMatchScheduler("Qwen-Image")`, like Qwen-Image-Layered: at inference `dynamic_shift_len = (H // 16) * (W // 16)` (or `exponential_shift_mu`), in training the template default (mu 0.8). Height and width division factor 16. Run these units in order:

1. **`LayerEditUnit_ShapeChecker`:** validates height and width.
2. **`LayerEditUnit_NoiseInitializer`:** generates noise of shape `(1, 16, 1 + layer_num, H // 8, W // 8)`.
3. **`LayerEditUnit_PromptEmbedder`:** reuses `QwenImageUnit_PromptEmbedder.encode_prompt`, with separate positive and negative prompts for CFG.
4. **`LayerEditUnit_LayerInputImageEmbedder`:** encodes `layer_input_image` with the RGBA VAE, following `diffsynth/pipelines/qwen_image.py`, and outputs `first_frame_latents`: the condition, and the target of frame 0.
5. **`LayerEditUnit_InputLayersEmbedder`:** builds `latents` (noise for all `1 + n` frames), and during training also `input_latents`, which is `[layer_input_image (composite) | n target layers]`.
6. **`LayerEditUnit_MaskEmbedder`:** builds `masks`, as described below.

### Mask unit

The `"mask"` key holds `list[Image (RGB/RGBA) | None]`, aligned with the `1 + n` frames: `mask[0]` is for `layer_input_image`, and `mask[1..n]` are for the layers.

1. Treat missing masks (`None`, or a list shorter than `1 + n`) as solid white. If there is no `mask` at all, the DiT uses all ones. A list longer than `1 + n` raises an error.
2. Convert to RGB, which drops alpha without compositing, then to luminance. This gives one single-channel tensor per frame in the closed range [0, 1], with white = 1 and black = 0.
3. Resize to pixel size, then downsample to the latent size with area interpolation.
4. Keep the single channel (`mask_dim` = 1), giving `(1, 1, 1 + n, h, w)`. Masks cached with the channel replicated (older caches, 4 identical channels) are averaged back to one channel by the DiT, so no re-caching is needed.
5. The DiT concatenates masks with latents on channels before patchify, and uses them to select tokens (see Sparse tokens).

The DiT is not loaded during data caching. In that case, read `mask_dim` from the default value in the `LayerEditDiT` signature.

The appended condition tokens always use the all-ones mask. `mask[0]` applies only to frame 0.

### Inference

`pipe(prompt, layer_input_image=..., layer_num=..., mask=..., ...)` fills latent cells outside the masks with the VAE encoding of a fully transparent image, skips frame 0 (the generated composite, like Qwen-Image-Layered) when decoding, and decodes each of the `n` layers separately. It returns a list of `n` RGBA images.

## Weight initialization

`dit_init.py` writes bf16 `.safetensors` weights to `models/LayerEdit/layer_edit_dit.safetensors` by default. Options: `--qwen_path`, `--output_path`, `--seed` (default 0).

Randomly initialize the whole model with `init_weights()` first, then copy every available weight. Parameters without a source keep their random values, and the log lists them (none with the current architecture).

| Source (Qwen-Image-Layered) | Target |
|---|---|
| `img_in` `[3072, 64]`, layout (C P Q) | `img_in` `[3072, 68]`, layout (P Q C): 16 latent channels permuted, 1 mask channel zero |
| `time_text_embed.timestep_embedder` | `time_embedding` |
| `time_text_embed.addition_t_embedding[0]` | Added into `time_embedding.2.bias` |
| `txt_norm`, `txt_in` | `text_embedding.0`, `text_embedding.1` |
| `transformer_blocks.j.<name>` | `blocks.j.<name>`, same name and shape, one-to-one |
| `transformer_blocks.j.txt_mod.1` `[18432, 3072]` | `blocks.j.txt_mod.1` `[6144, 3072]`: rows 0 to 6143, the (shift, scale) of the attention |
| `norm_out.linear` | `head.linear` |
| `proj_out` `[64, 3072]`, layout (C P Q) | `head.head`, layout (P Q C), permuted |

- Dropped source tensors: `txt_mlp`, the gate and MLP rows of `txt_mod`, `attn.add_q_proj`, `attn.norm_added_q`, `attn.to_add_out`.
- The 12,288 mask columns of `img_in.weight` (4 mask inputs per 2x2 patch x 3072 outputs) are zero, so masks change nothing at initialization. No weight stays random.
- The block count must match the source (60), otherwise raise an error.
- The one difference from Qwen that copying weights cannot remove, and fine-tuning must adapt to: from the second block on, Qwen's text has been updated by the blocks before, LayerEdit's has not.

After changing the architecture, rerun `dit_init.py` and update `model_hash` in `layer_edit_series` with the hash it prints. Its `extra_kwargs` are `{"dim": 3072, "ffn_dim": 12288, "num_heads": 24, "num_layers": 60, "cycle_length": 3}`.

## Training

### Dataset

`metadata.json` format:

```json
{"image": ["composite.png", "layer_1.png", "..."], "layer_num": 2, "layer_input_image": "composite.png", "mask": ["mask_0.png", "mask_1.png", "..."], "prompt": "..."}
```

`image[0]` is the composite, which is the same as `layer_input_image`, and `image[1:]` are the target RGBA layers. A list of only the `n` layers is also accepted. `mask` is optional and may contain `null` entries.

### Loss modes

`layer_edit_full.sh` selects the mode with one variable, `LOSS_TYPE`, passed to `--loss_type`. The script defaults to `flow_match`; `train.py` used directly defaults to `rgba`.

Both modes use `LayerEditRGBALoss`.

- **`flow_match`:** MSE on all `1 + n` frames (frame 0 = the composite, as Qwen-Image-Layered generates it), averaged over kept latent cells. Dropped tokens are excluded. Without `layer_input_image`, frame 0 has no target and is excluded.
- **`rgba`:** The same flow-matching loss on all parameters, plus RGB and alpha losses routed to specific blocks:
  1. Run the full DiT forward pass to get the final prediction `v`.
  2. Compute `x0 = x_t - sigma * v`, fill dropped cells from the target latents, and decode with the frozen RGBA VAE. Decode the target latents the same way. Only the layers (frames 1 to n), not the composite.
  3. Compute `alpha_loss` as the MSE of alpha in [0, 1], and `rgb_loss` as the squared RGB error weighted by the target alpha.
  4. Route the gradients to the blocks only, with the default weights below. Embeddings and the head get no gradient from these losses. Do not compute a loss on intermediate hidden states.

Default block weights, set by `--rgba_loss_blocks` as `first-last:rgb_weight:alpha_weight`. The pattern repeats A, B, B across the 60 blocks (3 x 20). An empty string disables these losses.

| Group | Blocks | RGB weight | Alpha weight |
|---|---|---|---|
| A | 0, 3, ..., 57 | 0.2 | 0.8 |
| B | All others | 0.8 | 0.2 |

The groups only route losses. A and B blocks have the same architecture.

**Gradient routing:** take the RGB and alpha gradients with respect to the output of each group's last block, using two `torch.autograd.grad` calls. Re-run each group on detached inputs and add the surrogate `sum(group_output * weighted_gradient)`. The reported loss value stays equal to the flow-matching loss; only the surrogate's gradient matters.

### Commands

Run from the repo root, in this order:

1. **Initialize weights:** run `dit_init.py`. This step is commented out in `layer_edit_full.sh`.
2. **Cache data:** `--task sft:data_process` loads the text encoder and VAE, with no DiT, and writes to `./models/train/LayerEdit_full-cache`.
3. **Train:** `--task sft:train` reads the cache as its dataset, loads the initial DiT weights and the VAE, and writes to `./models/train/LayerEdit_full_${LOSS_TYPE}`. Use `accelerate_config_14B.yaml`, with `--find_unused_parameters` and `--initialize_model_on_cpu`.

Current settings: `--max_pixels 100000`, `--learning_rate 1e-5`, `--num_epochs 5`, `--extra_inputs "layer_num,layer_input_image,mask"`, `--data_file_keys "image,layer_input_image,mask"`.

`accelerate_config_14B.yaml` (1 process, ZeRO-2, optimizer and parameters offloaded to CPU, bf16) is the recommended config. `accelerate_config_zero3.yaml` is less suitable for `rgba` because the routed loss re-runs blocks and calls `torch.autograd.grad` inside the step.

## Low-VRAM inference

`LayerEdit.py` reads samples from `--metadata_path`, runs the pipeline with disk offload, and writes `input.png`, `layer_N.png`, and `layer_N_gt.png` for each sample. Defaults: `--max_pixels 262144`, `--num_inference_steps 50`, `--cfg_scale 4.0`, `--seed 0`, `--onload_dtype float8_e4m3fn`. Masks are used by default, so inference is sparse and prints the kept tokens per sample; `--no-use_mask` sends all tokens to the DiT.
