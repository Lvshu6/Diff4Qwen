# LayerEditDiT: image -> Live2D layered PSD (full pre-training)
# Dataset metadata format (same as Qwen-Image-Layered, plus optional masks):
#   {"image": ["xxx/image.png", "xxx/image_1.png", ...], "layer_num": 3, "layer_input_image": "xxx/image.png",
#    "mask": [null, "xxx/mask_1.png", ...], "prompt": "..."}
#   image[0] is the composite (== layer_input_image), image[1:] are the target RGBA layers.
#   mask (optional) is aligned with the 1 + n frames: mask[0] -> layer_input_image, mask[1..n] -> the n layers.
#   Missing entries (null, or a shorter list) default to all white. Each mask is converted to a single channel
#   in [0, 1] (RGB luminance only, white = 1, black = 0), downsampled to the latent resolution, replicated to
#   4 channels and concatenated to the latents before patchify.
# Multi-GPU: --config_file examples/layer_edit/model_training/full/accelerate_config_14B.yaml (ZeRO-2 + CPU offload, recommended)
#            or accelerate_config_zero3.yaml (ZeRO-3; the block-routed RGB / alpha loss re-runs block groups
#            and calls torch.autograd.grad inside the step, which is less well supported by ZeRO-3)
# --max_pixels 1048576 \ 518400 262144
# Loss type (the only switch, set LOSS_TYPE below or `LOSS_TYPE=flow_match bash layer_edit_full.sh`):
#   flow_match: the common flow-matching loss only (same as examples/wanvideo/model_training/train.py)
#   rgba      : flow-matching loss + block-routed RGB / alpha losses, described below
LOSS_TYPE=${LOSS_TYPE:-flow_match}
# Losses of LOSS_TYPE=rgba (one training run, every iteration):
#   flow-matching loss: all parameters
#   RGB / alpha losses (VAE-decoded x0 prediction vs. GT RGBA), routed to specific blocks only:
#     DiT blocks follow the cyclic pattern (A, RGB, RGB) * 10:
#     A   blocks (0, 3, ..., 27; self + text attention)       : RGB 0.2, alpha 0.8
#     RGB blocks (the others; self-attention only)            : RGB 0.8, alpha 0.2
#   This is the default of --rgba_loss_blocks, i.e. "0:0.2:0.8,1-2:0.8:0.2,3:0.2:0.8,4-5:0.8:0.2,..." (empty string disables them)
# The RGB / alpha losses decode the predicted latents with the RGBA VAE, so the VAE is also loaded in the training step.

# Step 0: initialize the DiT weights (bf16, safetensors)
# python examples/layer_edit/model_training/dit_init.py \
#   --preset small \
#   --output_path "models/LayerEdit/layer_edit_dit_small.safetensors"

# Step 1: cache VAE latents (layers, layer_input_image, masks) and text embeddings
# accelerate launch examples/layer_edit/model_training/train.py \
#   --dataset_base_path data/combine \
#   --dataset_metadata_path data/combine/metadata.json \
#   --data_file_keys "image,layer_input_image,mask" \
#   --max_pixels 100000 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/LayerEdit_full-cache" \
#   --trainable_models "dit" \
#   --extra_inputs "layer_num,layer_input_image,mask" \
#   --use_gradient_checkpointing \
#   --dataset_num_workers 8 \
#   --task "sft:data_process" 
# > layer_edit_data_process.log 2>&1

# Step 2: training
nohup accelerate launch --config_file examples/layer_edit/model_training/full/accelerate_config_14B.yaml examples/layer_edit/model_training/train.py \
  --dataset_base_path "./models/train/LayerEdit_full-cache" \
  --data_file_keys "image,layer_input_image,mask" \
  --max_pixels 100000 \
  --dataset_repeat 1 \
  --model_paths '["models/LayerEdit/layer_edit_dit_large.safetensors"]' \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LayerEdit_full_${LOSS_TYPE}" \
  --trainable_models "dit" \
  --extra_inputs "layer_num,layer_input_image,mask" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --initialize_model_on_cpu \
  --loss_type "${LOSS_TYPE}" \
  --task "sft:train" \
> layer_edit_train_${LOSS_TYPE}.log 2>&1 &
