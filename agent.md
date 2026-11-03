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

Use the Qwen-Image-Layered DiT and remove only its text stream. Everything else keeps Qwen's widths, parameter names, and activations, so block weights can be copied by name.

Defaults: `dim=3072`, `ffn_dim=12288`, `num_heads=24` (head dim 128), `num_layers=60`, `in_dim=16`, `mask_dim=4`, `out_dim=16`, `text_dim=3584`, `freq_dim=256`, `patch_size=(1, 2, 2)`, `cycle_length=3`, `rope_max_len=1024`, `eps=1e-6`.

**Total: 11,368,349,760 parameters.**

### Text

- `text_embedding` is `RMSNorm(3584)` followed by `Linear(3584, dim)`, which correspond to Qwen's `txt_norm` and `txt_in`.
- No network layer updates the text tokens after `text_embedding`. The same tokens go to every block, and no block normalizes, modulates, or writes them back.
- Remove `txt_mod`, `txt_norm1`, `txt_norm2`, `txt_mlp`, `attn.add_q_proj`, `attn.norm_added_q`, and `attn.to_add_out`.
- Keep `attn.add_k_proj`, `attn.add_v_proj`, and `attn.norm_added_k`. They only derive keys and values for the attention in that block, and their output is discarded afterward.

### Blocks

All 60 `DiTBlock`s are identical and contain:

- **Timestep projection:** each block has its own `img_mod = Sequential(SiLU, Linear(dim, 6 * dim))`, as in Qwen. Do not share one projection across blocks.
- **Attention:** `JointAttention`. Only image tokens are queries. Keys and values are `[text, image]` concatenated on the sequence and share one softmax. RoPE applies to image queries and keys only; text gets none. Keep Qwen's per-head `RMSNorm` (`norm_q`, `norm_k`, `norm_added_k`).
- **MLP:** `img_mlp` with Qwen's approximate GELU, `x * sigmoid(1.702 * x)`. Parameter names are `net.0.proj` and `net.2`.
- **Norms:** `img_norm1` and `img_norm2`, `LayerNorm` without learnable parameters.

Other modules:

- `time_embedding`: sinusoidal embedding of `freq_dim`, then `Linear`, `SiLU`, `Linear`.
- `Head`: non-learnable `LayerNorm`, `linear = Linear(dim, 2 * dim)` applied to `silu(temb)` to give (scale, shift), and `head = Linear(dim, 64)`.
- RoPE: 3D axes of (44, 42, 42) for (layer, height, width), with positions counted from 0. The first axis indexes layer order.

### Layer sequence

Adapt video-generation logic to multiple layers by treating the time dimension as layer order. Latents have shape `(B, 16, 1 + n, h, w)`:

- Frame 0 is `layer_input_image`, the clean condition. It overwrites frame 0 of the latents, as in image-to-video generation, and is written again after every scheduler step.
- Frames 1 to n are the layers to generate, starting from noise appended after the condition.

### Input tokens

`patchify` has no parameters. It reshapes each 2x2 patch into the channel dimension: `(B, C, F, H, W)` to `(B, F * H/2 * W/2, 4 * C)`.

1. **Latent tokens:** concatenate latents `(B, 16, 1 + n, h, w)` and masks `(B, 4, 1 + n, h, w)` on channels, then patchify.
2. **Condition tokens:** take `first_frame_latents` of `layer_input_image` `(B, 16, 1, h, w)`, concatenate an all-ones mask `(B, 4, 1, h, w)` on channels, then patchify.
3. **Sequence:** concatenate latent tokens and condition tokens on the token sequence dimension, giving `(B, (1 + n + 1) * h/2 * w/2, 80)`.
4. **Projection:** apply `img_in = Linear(80, dim)`.

The condition image therefore appears twice: as overwritten frame 0 and as the appended condition tokens. The appended tokens take RoPE layer index `1 + n`.

### Timesteps and output

- Compute `temb` for two values, `[noisy timestep, 0]`. Each block projects both through its own `img_mod`, then selects per token.
- Frame 0 and the appended condition tokens use timestep 0. Frames 1 to n use the noisy timestep.
- Drop the appended condition tokens before `Head`, then unpatchify. The output is `(B, 16, 1 + n, h, w)`, in the same frame order as the input latents.

## Pipeline units

`LayerEditPipeline` uses `FlowMatchScheduler("Wan")`, `sigma_shift=5.0`, and a height and width division factor of 16. Run these units in order:

1. **`LayerEditUnit_ShapeChecker`:** validates height and width.
2. **`LayerEditUnit_NoiseInitializer`:** generates noise of shape `(1, 16, 1 + layer_num, H // 8, W // 8)`.
3. **`LayerEditUnit_PromptEmbedder`:** reuses `QwenImageUnit_PromptEmbedder.encode_prompt`, with separate positive and negative prompts for CFG.
4. **`LayerEditUnit_LayerInputImageEmbedder`:** encodes `layer_input_image` separately with the RGBA VAE as an additional condition, following `diffsynth/pipelines/qwen_image.py`, and outputs `first_frame_latents`.
5. **`LayerEditUnit_InputLayersEmbedder`:** builds `latents`, and during training also `input_latents`, which is `[layer_input_image | n target layers]`.
6. **`LayerEditUnit_MaskEmbedder`:** builds `masks`, as described below.

### Mask unit

The `"mask"` key holds `list[Image (RGB/RGBA) | None]`, aligned with the `1 + n` frames: `mask[0]` is for `layer_input_image`, and `mask[1..n]` are for the layers.

1. Treat missing masks (`None`, or a list shorter than `1 + n`) as solid white. If there is no `mask` at all, the DiT uses all ones. A list longer than `1 + n` raises an error.
2. Convert to RGB, which drops alpha without compositing, then to luminance. This gives one single-channel tensor per frame in the closed range [0, 1], with white = 1 and black = 0.
3. Resize to pixel size, then downsample to the latent size with area interpolation.
4. Replicate the single channel to `mask_dim` (4) channels, giving `(1, 4, 1 + n, h, w)`.
5. The DiT concatenates masks with latents on channels before patchify.

The DiT is not loaded during data caching. In that case, read `mask_dim` from the default value in the `LayerEditDiT` signature.

The appended condition tokens always use the all-ones mask. `mask[0]` applies only to frame 0.

### Inference

`pipe(prompt, layer_input_image=..., layer_num=..., mask=..., ...)` skips the condition frame when decoding, and decodes each of the `n` layers separately. It returns a list of `n` RGBA images.

## Weight initialization

`dit_init.py` writes bf16 `.safetensors` weights to `models/LayerEdit/layer_edit_dit.safetensors` by default. Options: `--qwen_path`, `--output_path`, `--seed` (default 0).

The mask design differs from Qwen-Image-Layered. Randomly initialize the whole model with `init_weights()` first, then copy every available weight. Parameters without a source keep their random values, and the log lists them.

| Source (Qwen-Image-Layered) | Target |
|---|---|
| `img_in` `[3072, 64]`, layout (C P Q) | `img_in` `[3072, 80]`, layout (P Q C): 16 latent channels permuted, 4 mask channels stay random |
| `time_text_embed.timestep_embedder` | `time_embedding` |
| `time_text_embed.addition_t_embedding[0]` | Added into `time_embedding.2.bias` |
| `txt_norm`, `txt_in` | `text_embedding.0`, `text_embedding.1` |
| `transformer_blocks.j.<name>` | `blocks.j.<name>`, same name and shape, one-to-one |
| `norm_out.linear` | `head.linear` |
| `proj_out` `[64, 3072]`, layout (C P Q) | `head.head`, layout (P Q C), permuted |

- Dropped source tensors: `txt_mod`, `txt_mlp`, `attn.add_q_proj`, `attn.norm_added_q`, `attn.to_add_out`.
- The only random weights after copying are the 49,152 mask columns of `img_in.weight` (16 mask inputs x 3072 outputs).
- The block count must match the source (60), otherwise raise an error.
- These differences from Qwen cannot be removed by copying weights, and fine-tuning must adapt to them: the RoPE axes and positions differ, and text keys and values differ because the text is no longer modulated or updated.

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

- **`flow_match`:** `FlowMatchSFTLoss`. MSE on frames 1 to n; the condition frame is excluded.
- **`rgba`:** `LayerEditRGBALoss`. The same flow-matching loss on all parameters, plus RGB and alpha losses routed to specific blocks:
  1. Run the full DiT forward pass to get the final prediction `v`.
  2. Compute `x0 = x_t - sigma * v` and decode it with the frozen RGBA VAE. Decode the target latents the same way. Exclude the condition frame.
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

`LayerEdit.py` reads samples from `--metadata_path`, runs the pipeline with disk offload, and writes `input.png`, `layer_N.png`, and `layer_N_gt.png` for each sample. Defaults: `--max_pixels 262144`, `--num_inference_steps 50`, `--cfg_scale 4.0`, `--seed 0`, `--onload_dtype float8_e4m3fn`. Masks are used only with `--use_mask`.
