Create new model architecture: LayerEditDit for image‑to‑Live2D‑layer PSD segmentation‑and‑generation tasks.

`diffsynth/models/wan_video_dit.py`: Borrow the Fused‑overwrite DiT design from wan_video. At inference time, take `layer_input_image` as input, append pending‑generation noise analogous to image‑to‑video generation. Adapt video‑generation logic for multi‑layer generation; repurpose the time dimension as layer order, yielding 1+n entries (1 stands for overwrite `layer_input_image`; n denotes segmented multiple layers). Produce a clean self‑contained script `diffsynth/models/layer_edit_dit.py` by stripping out superfluous plugin units: vace, controlnet, audio, animate, wan2dance and others. Note patchify uses parameter‑free 2×2‑patch‑to‑channel tensor reshaping; implement `dit.img_in` to permute dimensions for DiT.

`diffsynth/models/qwen_image_vae.py`: Follow RGBA‑VAE reference with `image_channel=4`. Directly reuse the RGBA VAE from qwen_image_layer.

`diffsynth/models/qwen_image_text_encoder.py`: Directly reuse the text encoder from qwen_image_layer.

For model pre‑training, create weight‑initialization script `dit_init.py` with bf16 precision and `.safetensors` output format.

Generate `examples/layer_edit/model_training/train.py` by referencing `examples/wanvideo/model_training/train.py`.

Generate `examples/layer_edit/model_training/layer_edit_full.sh` by referencing `examples/qwen_image/model_training/Qwen‑Image‑Layered_distill.sh`.

Copy `examples/wanvideo/model_training/full/accelerate_config_14B.yaml` and `examples/wanvideo/model_training/full/accelerate_config_zero3.yaml` into the layer_edit directory.

Model details:
Unit design: key `"mask"`: `list[Image:RGB/RGBA]`, aligned with the 1+n sequence; missing entries default to all‑white. Extract only RGB values: white maps to 1, black maps to 0. Convert via RGB luminance into list of single‑channel values within range (0, 1). Downsample to latent resolution, replicate the single channel into four channels. Concatenate mask onto latent tensor dimensions **before patchify**, then run patchify to get `x`.

Refer to `diffsynth/pipelines/qwen_image.py` to process `layerinputimage`: encode it independently with VAE as an extra condition; its mask defaults to all‑ones for channel alignment.

Design training logic following `examples/wanvideo/model_training/train.py`.

Add one constraint: VAE‑decode predicted training RGBA images, compute RGB loss and Alpha(A) loss. Apply this constraint only to designated DiT layers; RGB‑&‑Alpha losses shall update only target layers and leave others untouched. Do not expose DiT intermediate outputs. Compute loss after VAE decoding the **final DiT latent output**, and constrain gradients exclusively for selected DiT layers.

Objectives:
1. Full DiT forward pass to produce final latent output.
2. VAE decode DiT final‑output latents to obtain predicted RGBA image.
3. Decompose predicted RGBA and ground‑truth RGBA respectively; compute RGB loss plus Alpha(A) loss.
4. Back‑propagate this composite loss to update parameters of specified DiT blocks only; remaining DiT layers receive zero gradient from this RGBA/Alpha loss.

Assign heterogeneous losses across DiT blocks. Keep the global flow‑match loss intact. Extra constraints: Block 1 (RGB loss 0.2, A loss 0.8); Blocks 2/3 (RGB loss 0.8, A loss 0.2). Enable heterogeneous training within a single training iteration: Block 1 (RGB 0.2 / A 0.8) + Blocks 2/3 (RGB 0.8 / A 0.2). Build a 30‑layer DiT as ten repetitions of a three‑layer cycle following pattern A, RGB, RGB. Inside each triplet: Layer 1 carries self‑attention + text‑attention; Layers 2 and 3 carry self‑attention only.