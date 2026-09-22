# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image/*" --local_dir ./data/diffsynth_example_dataset
# add:model_paths, "layer_split_image" trainable_models "dit" \

# accelerate launch examples/qwen_image/model_training/train.py \
#   --dataset_base_path data/psd \
#   --dataset_metadata_path data/psd/metadata.json \
#   --data_file_keys "image,layer_input_image,layer_split_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 10 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit.,pipe.layer_adapter." \
#   --output_path "./models/train/Qwen-Image-Layered_lora-splited-cache" \
#   --trainable_models "layer_adapter" \
#   --lora_base_model "dit" \
#   --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
#   --lora_rank 32 \
#   --extra_inputs "layer_num,layer_input_image,layer_split_image" \
#   --use_gradient_checkpointing \
#   --dataset_num_workers 8 \
#   --find_unused_parameters \
#   --task "sft:data_process"

accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "./models/train/Qwen-Image-Layered_lora-splited-cache" \
  --data_file_keys "image,layer_input_image,layer_split_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths '["./models/LayerAdapter/layer_adapter.safetensors"]' \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Layered:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit.,pipe.layer_adapter." \
  --output_path "./models/train/Qwen-Image-Layered_lora" \
  --trainable_models "layer_adapter" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --extra_inputs "layer_num,layer_input_image,layer_split_image" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --initialize_model_on_cpu \
  --task "sft:train"
