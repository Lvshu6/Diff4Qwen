Create new model architecture: LayerEditDiT for image‑to‑Live2D layered PSD segmentation‑and‑generation task.

`diffsynth/models/wan_video_dit.py`: Adopt Wan‑Video’s fused‑overwrite DiT design. At inference, input `layer_input_image`. Append to‑be‑generated noise afterwards like image‑to‑video. Repurpose video generation for multi‑layer generation; the time dimension becomes layer order. There are `1+n` frames: 1 for overwrite `layer_input_image`, n for segmented output layers. Create a clean complete script `diffsynth/models/layer_edit_dit.py`, removing unused units such as VACE, ControlNet, audio, animate, wan2dance. Patchify uses parameter‑free 2×2 patch flatten‑to‑channel tensor rearrangement. Design `dit.img_in` for dimension transformation.

`diffsynth/models/qwen_image_vae.py`: Reuse RGBA‑VAE with `image_channel=4` from Qwen‑Image‑Layered.
`diffsynth/models/qwen_image_text_encoder.py`: Reuse text encoder from Qwen‑Image‑Layered.

Implement weight‑initialization script `dit_init.py` for pretraining; output bf16 `.safetensors`.

Port `examples/wanvideo/model_training/train.py` to `examples/layer_edit/model_training/train.py`.
Port `examples/qwen_image/model_training/Qwen‑Image‑Layered_distill.sh` to `examples/layer_edit/model_training/layer_edit_full.sh`.
Copy `accelerate_config_14B.yaml` and `accelerate_config_zero3.yaml` from `examples/wanvideo/model_training/full/` to layer_edit folder.

Model details for Unit design:
Key `mask`: list[RGB/RGBA Image], aligned with `1+n` frames. Missing entries default to solid‑white. Extract only RGB values; convert luminance to single‑channel tensors in range `[0,1]` (white=1, black=0). Downsample to latent resolution, replicate single channel to 4 channels. Concatenate mask with latents along channel dimension **before patchify**, then perform patchify to get `x`.

Follow `diffsynth/pipelines/qwen_image.py` to process `layer_input_image`: encode independently via VAE as additional condition. Its mask defaults to all‑ones for channel alignment.

Implement training logic referencing `examples/wanvideo/model_training/train.py`.

Add constraint: decode DiT predicted latents through VAE to obtain predicted RGBA images; compute RGB loss and alpha loss. Apply these losses only to selected DiT blocks, leaving other blocks unaffected. Loss is computed on DiT’s final output after VAE decoding, not on intermediate DiT hidden states.

Objectives:
1. Full DiT forward pass yields final latent output.
2. VAE decodes final DiT latents into predicted RGBA images.
3. Split predicted and ground‑truth RGBA, compute RGB loss and alpha loss.
4. Back‑propagate these losses **only to designated DiT blocks; other DiT blocks receive no gradients from RGBA/alpha loss**.

Differentiated loss assignment across DiT blocks: global flow‑match loss remains unchanged. Extra constraints: block‑1 uses RGB weight 0.2 / alpha weight 0.8; block‑2/3 use RGB weight 0.8 / alpha weight 0.2. Within one training step realize cyclic pattern `A, RGB, RGB` repeated 10 times for total 30 DiT blocks. Block‑type‑A contains self‑attention + text‑attention; block‑type‑RGB contains only self‑attention.

Two loss training modes: the standard `flow_match` from `examples/wanvideo/model_training/train.py`, plus the above RGBA‑block‑routed loss. Switch via one shell parameter.

For `first_frame_latent` (VAE‑encoded `layer_input_image`): attach all‑one mask channels, apply rearrange, concatenate with rearranged latents on channel dimension, then feed into `img_in` projection.