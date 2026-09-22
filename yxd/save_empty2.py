import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from einops import rearrange
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors
from diffsynth.models.qwen_image_layer_adapter import QwenImageLayerAdapter


def save_state_dict(
    state_dict: dict,
    file_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device: str = "cpu"
):
    processed_state = {}
    for k, v in state_dict.items():
        if torch_dtype is not None:
            v = v.clone().to(dtype=torch_dtype)
        if device is not None:
            v = v.to(device=device)
        processed_state[k] = v

    if file_path.endswith(".safetensors"):
        save_safetensors(processed_state, file_path)
        print(f"权重已以 {torch_dtype} 格式保存为 safetensors：{file_path}")
    elif file_path.endswith(".bin"):
        torch.save(processed_state, file_path)
        print(f"权重已以 {torch_dtype} 格式保存为 bin：{file_path}")
    else:
        raise ValueError("文件后缀必须是 .safetensors 或 .bin")


def load_state_dict(
    file_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device: str = "cpu"
):
    if file_path.endswith(".safetensors"):
        state_dict = load_safetensors(file_path, device=device)
        if torch_dtype is not None:
            state_dict = {k: v.clone().to(dtype=torch_dtype) for k, v in state_dict.items()}
        return state_dict
    elif file_path.endswith(".bin"):
        map_location = torch.device(device)
        state_dict = torch.load(
            file_path,
            map_location=map_location,
            weights_only=True
        )
        if torch_dtype is not None:
            state_dict = {k: v.clone().to(dtype=torch_dtype) for k, v in state_dict.items()}
        return state_dict
    else:
        raise ValueError("文件后缀必须是 .safetensors 或 .bin")


if __name__ == "__main__":
    model = QwenImageLayerAdapter()
    print("原始模型权重 dtype：", model.layer_adapter.fuser_blocks[0].linear1_kv.weight.dtype)

    safetensors_bf16_path = "./models/LayerAdapter/layer_adapter.safetensors"
    save_state_dict(
        state_dict=model.state_dict(),
        file_path=safetensors_bf16_path,
        torch_dtype=torch.bfloat16,
        device="cpu"
    )

    model = model.to(dtype=torch.bfloat16)
    print("模型转换为 BFloat16 后的 dtype：", model.layer_adapter.fuser_blocks[0].linear1_kv.weight.dtype)

    loaded_bf16_safe_dict = load_state_dict(
        safetensors_bf16_path,
        torch_dtype=torch.bfloat16,
        device="cpu"
    )
    model.load_state_dict(loaded_bf16_safe_dict)
    print("加载后模型权重 dtype：", model.layer_adapter.fuser_blocks[0].linear1_kv.bias.dtype)
    print("BFloat16 Safetensors 权重加载成功")
