# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image/*" --local_dir ./data/diffsynth_example_dataset
# add:model_paths, "layer_split_image" trainable_models "dit" \
#   --max_pixels 1048576 \ 518400 262144
# --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml

# nohup accelerate launch examples/qwen_image/model_training/train.py \
#   --dataset_base_path data/parquet_data/export \
#   --dataset_metadata_path data/parquet_data/export.json \
#   --data_file_keys "image,layer_input_image" \
#   --max_pixels 262144 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Qwen-Image-Layered_distill-splited-cache" \
#   --trainable_models "dit" \
#   --extra_inputs "layer_num,layer_input_image" \
#   --use_gradient_checkpointing \
#   --dataset_num_workers 8 \
#   --find_unused_parameters \
#   --task "sft:data_process" \
# > train2.log 2>&1 &


nohup accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "./models/train/Qwen-Image-Layered_distill-splited-cache" \
  --data_file_keys "image,layer_input_image" \
  --max_pixels 262144 \
  --dataset_repeat 1 \
  --model_paths '["models/Student/qwen_image_layered30.safetensors"]' \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Layered_distill" \
  --trainable_models "dit" \
  --extra_inputs "layer_num,layer_input_image" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --initialize_model_on_cpu \
  --task "sft:train" \
> train3.log 2>&1 &
