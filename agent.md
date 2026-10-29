Create the new model architecture `LayerEditDit` for the image‑to‑Live2D layered‑PSD segmentation‑generation task.

`diffsynth/models/wan_video_dit.py`: Adopt the fused‑overwrite DiT design from wan‑video. At inference, take `layer_input_image` as input and concatenate pending‑generation noise afterwards following the image‑to‑video paradigm. Migrate video‑generation logic to multi‑layer generation: repurpose the time dimension as layer order to form a `1+n` sequence — `1` corresponds to the overwrite `layer_input_image`, and `n` corresponds to segmented target layers. Produce a clean standalone script `diffsynth/models/layer_edit_dit.py` by stripping auxiliary plugin units including VACE, ControlNet, audio, animate, and wan2dance. Implement parameter‑free patchify via tensor reshaping that flattens 2×2 patches into the channel dimension. Design `dit.img_in` to project reshaped tensors into DiT‑compatible dimensions.

`diffsynth/models/qwen_image_vae.py`: Reference the RGBA‑capable VAE with `image_channel=4`. Directly reuse the RGBA VAE from qwen_image_layer.

`diffsynth/models/qwen_image_text_encoder.py`: Directly reuse the text encoder from qwen_image_layer.

For pretraining, create the weight‑initialization script `dit_init.py` that outputs BF16 `.safetensors` weights.

Port `examples/wanvideo/model_training/train.py` to `examples/layer_edit/model_training/train.py`.
Port `examples/qwen_image/model_training/Qwen‑Image‑Layered_distill.sh` to `examples/layer_edit/model_training/layer_edit_full.sh`.
Copy `examples/wanvideo/model_training/full/accelerate_config_14B.yaml` and `examples/wanvideo/model_training/full/accelerate_config_zero3.yaml` into the layer_edit directory.

Model details:
Unit specification: key `"mask"` accepts `list[Image(RGB/RGBA)]` aligned with the `1+n` sequence. Missing or null entries default to solid‑white masks. Extract RGB‑luminance values, map white to 1 and black to 0 to produce single‑channel masks in the range `[0,1]`. Resize masks to latent resolution, replicate the single channel into 4 channels. Concatenate mask channels to latent tensors **before patchify**, then run patchify to obtain tensor `x`.

Follow `diffsynth/pipelines/qwen_image.py` to separately VAE‑encode `layer_input_image` as an additional condition; its associated mask defaults to all‑ones for channel alignment.
Follow the training workflow defined in `examples/wanvideo/model_training/train.py`.

Implement the following constraint: decode the DiT’s final latent output through VAE to get predicted RGBA images, then compute RGB loss and Alpha loss. This loss must apply gradients only to designated DiT blocks; other blocks receive zero gradient from this RGB‑Alpha loss. Do not tap intermediate DiT outputs; compute loss purely from VAE‑decoded final predictions.

Execution pipeline for the RGBA‑Alpha auxiliary loss:
1. Run full DiT forward pass to obtain the final latent output.
2. VAE‑decode the final latent to get the predicted RGBA image.
3. Split predicted RGBA and ground‑truth RGBA, compute RGB loss plus Alpha loss.
4. Back‑propagate this composite loss only to specified DiT blocks; suppress gradients for all other DiT blocks against this loss signal.

Keep the global flow‑match loss unchanged while applying differentiated loss‑weight assignment across DiT blocks:
‑ Group‑A blocks: RGB loss weight = 0.2, Alpha loss weight = 0.8
‑ Group‑B blocks: RGB loss weight = 0.8, Alpha loss weight = 0.2

Cycle this triplet pattern for a total of 36 DiT blocks (`3 × 12`), following the repeating sequence `A, B, B, A, B, B …`.
Within each triplet:
‑ Group‑A block: self‑attention + text‑attention
‑ Two Group‑B blocks: self‑attention only

Support two training‑loss modes: the original flow‑match loss, and the above RGBA‑Alpha auxiliary loss scheme. Toggle modes via one shell‑script parameter.

Append an all‑ones mask channel to the VAE‑encoded `first_frame_latent` of `layer_input_image`, perform reshaping, concatenate with other latents along the sequence‑token dimension, then feed into the `img_in` projection layer.

Without modifying code, tell me what minimum configuration changes are required for from‑scratch pretraining of the 14B model on a single GPU with a single micro‑batch.