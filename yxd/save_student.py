# -*- coding: utf-8 -*-
"""
初始化Qwen‑Image‑Layered学生模型：每2层取后一层(1,3,5...)，层数减半，bf16，保存safetensors
说明：教师N层，学生取 [1,3,5,...] 奇数索引block，学生层数 = num_base_blocks // 2
依赖：torch, safetensors, transformers, diffsynth
"""
import torch
import os
from safetensors.torch import save_file
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

# ===================== 配置区 =====================
OUTPUT_SAVE_DIR = "./qwen_image_layered_student_half_odd"  # 输出保存目录
TARGET_DTYPE = torch.bfloat16
SAMPLE_STEP = 2          # 每2层采样1层，取后一层（奇数索引 1,3,5...）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "Qwen/Qwen-Image-Layered"
# ==================================================

def main():
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)

    # 1. 加载教师DiT模型
    print("Loading teacher Qwen‑Image‑Layered DiT ...")
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=TARGET_DTYPE,
        device=DEVICE,
        model_configs=[
            ModelConfig(
                model_id=BASE_MODEL_ID,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"
            )
        ],
        vram_limit=24.0
    )
    teacher_dit = pipe.dit
    teacher_sd = teacher_dit.state_dict()
    num_teacher_blocks = len(teacher_dit.transformer_blocks)
    print(f"Teacher transformer_blocks total: {num_teacher_blocks}")

    # ========== 关键：start=1，取1,3,5,...每2层的后一层 ==========
    selected_src_block_idx = list(range(1, num_teacher_blocks, SAMPLE_STEP))
    num_student_blocks = len(selected_src_block_idx)
    print(f"Sample step={SAMPLE_STEP}, select source block indices: {selected_src_block_idx}")
    print(f"Student model total transformer blocks: {num_student_blocks}")

    student_sd = {}
    # 复制全部非transformer_blocks权重（嵌入、time_embed、norm_out、proj_out等完整保留）
    for key, val in teacher_sd.items():
        if not key.startswith("transformer_blocks."):
            student_sd[key] = val.detach().clone().to(dtype=TARGET_DTYPE).cpu()

    # 将选中的教师奇数block映射到学生连续序号 0,1,2...
    for student_blk_id, src_blk_id in enumerate(selected_src_block_idx):
        src_prefix = f"transformer_blocks.{src_blk_id}"
        dst_prefix = f"transformer_blocks.{student_blk_id}"
        for key, val in teacher_sd.items():
            if key.startswith(src_prefix + "."):
                new_key = key.replace(src_prefix, dst_prefix)
                student_sd[new_key] = val.detach().clone().to(dtype=TARGET_DTYPE).cpu()

    # 保存 safetensors
    out_safetensors = os.path.join(OUTPUT_SAVE_DIR, "diffusion_pytorch_model.safetensors")
    save_file(student_sd, out_safetensors)
    print(f"\nSaved student DiT (bf16, half‑layers, take odd index 1,3,5…): {out_safetensors}")

    # 输出配置json，记录采样信息
    import json
    config_json = {
        "_name_or_path": "qwen_image_layered_student_half_odd",
        "torch_dtype": "bfloat16",
        "sample_step": SAMPLE_STEP,
        "sample_start_index": 1,
        "teacher_num_blocks": num_teacher_blocks,
        "student_num_blocks": num_student_blocks,
        "selected_source_block_indices": selected_src_block_idx
    }
    config_path = os.path.join(OUTPUT_SAVE_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)
    print(f"Config json saved: {config_path}")

    print("\n==== Finished ====")
    print(f"Teacher blocks:{num_teacher_blocks} → Student blocks:{num_student_blocks}")


if __name__ == "__main__":
    main()
