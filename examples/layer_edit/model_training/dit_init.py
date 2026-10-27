"""
Initialize the LayerEditDiT weights for pre-training (bf16, .safetensors).

Usage:
    python examples/layer_edit/model_training/dit_init.py --preset small --output_path models/LayerEdit/layer_edit_dit_small.safetensors

Optional: `--init_from` copies every tensor whose name and shape match from an existing
Wan-Video DiT checkpoint (e.g. Wan2.1-T2V-1.3B for the small preset). Tensors that do not
match (e.g. text_embedding.0, since Qwen-Image text features are 3584-d, and img_in, which does not
exist in Wan) keep the random init. Wan also carries cross_attn weights in every block; those of
the RGB blocks are simply ignored because they only have self-attention.

The 30 blocks follow the cycle (A, RGB, RGB) * 10: A blocks (0, 3, ..., 27) have self + text
attention, RGB blocks only have self-attention.
"""
import torch, os, json, argparse
from safetensors.torch import save_file
from diffsynth.core.loader import load_state_dict, hash_model_file
from diffsynth.models.layer_edit_dit import LayerEditDiT, LAYER_EDIT_DIT_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize LayerEditDiT weights.")
    parser.add_argument("--preset", type=str, default="large", choices=list(LAYER_EDIT_DIT_PRESETS.keys()), help="Model size.")
    parser.add_argument("--output_path", type=str, default=None, help="Output .safetensors path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device used for initialization.")
    parser.add_argument("--init_from", type=str, default=None, help="Optional. Comma-separated Wan-Video DiT checkpoint files to copy shape-matching tensors from.")
    return parser.parse_args()


def copy_matching_tensors(model: LayerEditDiT, paths: list[str]):
    source = load_state_dict(paths, torch_dtype=torch.float32)
    target = model.state_dict()
    copied, skipped = [], []
    for name, value in target.items():
        if name in source and tuple(source[name].shape) == tuple(value.shape):
            value.copy_(source[name].to(value.device))
            copied.append(name)
        else:
            skipped.append(name)
    print(f"Copied {len(copied)} tensors from {paths}. {len(skipped)} tensors keep the random init.")
    if len(skipped) > 0:
        print("Randomly initialized tensors:", skipped[:20], "..." if len(skipped) > 20 else "")


def main():
    args = parse_args()
    output_path = args.output_path or f"models/LayerEdit/layer_edit_dit_{args.preset}.safetensors"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    config = LAYER_EDIT_DIT_PRESETS[args.preset]
    print(f"Initializing LayerEditDiT ({args.preset}): {config}")
    with torch.device(args.device):
        model = LayerEditDiT(**config)
    model.init_weights()
    if args.init_from is not None:
        with torch.no_grad():
            copy_matching_tensors(model, args.init_from.split(","))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e9:.3f}B")

    state_dict = {name: value.detach().to(dtype=torch.bfloat16, device="cpu").contiguous() for name, value in model.state_dict().items()}
    del model
    save_file(state_dict, output_path, metadata={"format": "pt", "model": "LayerEditDiT", "preset": args.preset, "config": json.dumps(config)})
    print(f"Saved: {output_path}")
    print(f"Model hash: {hash_model_file(output_path)} (registered in diffsynth/configs/model_configs.py -> layer_edit_series)")


if __name__ == "__main__":
    main()
