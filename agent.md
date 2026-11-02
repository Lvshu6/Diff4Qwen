Create new model architecture: LayerEditDit for image‑to‑Live2D‑layer PSD segmentation‑generation tasks.

Modify `layer_edit_dit`: remove only the complete text stream from `qwenimagelayer`. Text‑token vectors are never updated by any network layer after passing through `text_embedding`. Weights are initialized from `qwenimagelayer` checkpoints. Restore Qwen’s original per‑block independent timestep projection layers.

At inference, input is `layer_input_image`. Append target‑to‑generate noise afterwards similar to video generation. Adapt video‑generation logic to multi‑layer generation: treat the time dimension as layer order, for 1+n layers (1 for overwriting `layer_input_image`, n for segmented output layers).

`diffsynth/models/qwen_image_vae.py`: reference the RGBA VAE with `image_channel=4`. Directly reuse the RGBA VAE from `qwen_image_layer`.

`diffsynth/models/qwen_image_text_encoder.py`: directly reuse the text‑encoder from `qwen_image_layer`.

Create weight‑initialization script `dit_init.py`; output bf16 `.safetensors` weights initialized from `qwenimagelayer`.

`examples/layer_edit/model_training/train.py`
Reference `examples/qwen_image/model_training/Qwen‑Image‑Layered_distill.sh` to generate `examples/layer_edit/model_training/layer_edit_full.sh`.

Copy `examples/qwen/model_training/full/accelerate_config_14B.yaml` and `examples/qwen/model_training/full/accelerate_config_zero3.yaml` to the layer_edit directory.

Model details:
Unit specification: Key `"mask"` holds `list[Image: RGB/RGBA]` aligned to the 1+n layer count. Missing masks default to solid‑white. Extract only RGB values: map white to 1 and black to 0, convert RGB luminance into a list of single‑channel values in range (0, 1). Downsample to latent size; replicate the single channel into four channels. Prior to patchify, concatenate this mask with latent → patchify → x.

Refer to `diffsynth/pipelines/qwen_image.py` for processing `layer_input_image`: encode it separately via VAE as an additional condition; its default mask is all‑ones for channel alignment.

The VAE‑encoded `first_frame_latent` of `layer_input_image` also appends an all‑ones mask channel, undergoes rerange transformation, then concatenates with reranged latents along the token sequence dimension before projection through `img_in`.

Mask design differs from `qwenimagelayer`. Copy available weights during initialization; randomly initialize remaining missing parameters.