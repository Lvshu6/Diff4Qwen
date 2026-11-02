"""
Initialize the LayerEditDiT weights from the Qwen-Image-Layered DiT for pre-training (bf16, .safetensors).

Usage:
    python examples/layer_edit/model_training/dit_init.py \
        --output_path models/LayerEdit/layer_edit_dit.safetensors

Source: Qwen-Image-Layered, MMDiT with 60 dual-stream blocks, dim 3072, 24 heads x 128, MLP 12288.
Target: LayerEditDiT, the same blocks without the text stream (the text tokens from text_embedding are
        never updated). Every block parameter has the same name and shape as in the source, so the blocks
        are copied as they are. The mask design differs from Qwen-Image-Layered: parameters without a
        counterpart are randomly initialized (LayerEditDiT.init_weights).

Mapping (source -> target):
    img_in [3072, 64] (C P Q)        -> img_in [3072, 80] (P Q C), 16 latent channels permuted, 4 mask channels random
    time_text_embed.timestep_embedder -> time_embedding (same sinusoidal embedding, t in [0, 1000], cos first)
    addition_t_embedding[0]          -> folded into time_embedding.2.bias (Qwen-Image-Layered always uses index 0)
    txt_norm, txt_in                 -> text_embedding.0, text_embedding.1
    transformer_blocks.j.<name>      -> blocks.j.<name> (img_mod, attn.to_q/k/v/to_out, attn.norm_q/k,
                                        attn.add_k/v_proj, attn.norm_added_k, img_mlp)
    norm_out.linear                  -> head.linear
    proj_out [64, 3072] (C P Q)      -> head.head [64, 3072] (P Q C)

Dropped (the text stream): txt_mod, txt_mlp, attn.add_q_proj, attn.norm_added_q, attn.to_add_out.

Randomly initialized (no counterpart): the mask columns of img_in. Any other parameter that is missing
in the source is also kept at its random initialization and listed in the log.

Not representable without changing the architecture (fine-tuning has to adapt them):
    - RoPE: Qwen uses axes (16, 56, 56) with centered positions, LayerEditDiT uses (44, 42, 42) from 0,
      and the text tokens get no RoPE.
    - The text tokens are neither updated nor modulated (txt_norm1 + txt_mod) by the blocks, so the text
      keys / values differ from Qwen in every block. The image stream of a block is otherwise exact.
"""
import torch, os, re, glob, json, argparse
from safetensors import safe_open
from safetensors.torch import save_file
from diffsynth.core.loader import hash_model_file
from diffsynth.models.layer_edit_dit import LayerEditDiT


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize LayerEditDiT weights from Qwen-Image-Layered.")
    parser.add_argument("--qwen_path", type=str, default="models/Qwen/Qwen-Image-Layered/transformer", help="Folder or comma-separated safetensors files of the Qwen-Image-Layered DiT.")
    parser.add_argument("--output_path", type=str, default="models/LayerEdit/layer_edit_dit.safetensors", help="Output .safetensors path.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random initialization of the parameters without a counterpart.")
    return parser.parse_args()


class QwenSource:
    def __init__(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if os.path.isdir(path) else path.split(",")
        self.index = {}
        for file in files:
            handle = safe_open(file, framework="pt", device="cpu")
            for key in handle.keys():
                self.index[key] = handle
        self.num_blocks = 1 + max(int(m.group(1)) for k in self.index if (m := re.match(r"transformer_blocks\.(\d+)\.", k)))
        self.used = set()

    def __contains__(self, key):
        return key in self.index

    def __call__(self, key):
        self.used.add(key)
        return self.index[key].get_tensor(key).float()


def permute_img_in(weight, init, in_dim=16, mask_dim=4):
    # Qwen patch features are (C P Q), LayerEditDiT patchify gives (P Q C) with C = in_dim + mask_dim (p = 1 for the layer axis).
    # Mask channels keep the columns of `init` (random), they have no counterpart in Qwen-Image-Layered.
    out = init.float().clone()
    for q in range(2):
        for r in range(2):
            for c in range(in_dim):
                out[:, (q * 2 + r) * (in_dim + mask_dim) + c] = weight[:, c * 4 + q * 2 + r]
    return out


def permute_head(weight, out_dim=16):
    # Qwen output features are (C P Q), LayerEditDiT unpatchify expects (P Q C).
    out = torch.empty_like(weight)
    for q in range(2):
        for r in range(2):
            for c in range(out_dim):
                out[(q * 2 + r) * out_dim + c] = weight[c * 4 + q * 2 + r]
    return out


def init_from_qwen_layered(model: LayerEditDiT, source: QwenSource):
    params = dict(model.named_parameters())
    copied = set()

    def copy(name, value):
        param = params[name]
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(f"{name}: shape {tuple(value.shape)} does not match {tuple(param.shape)}.")
        param.data.copy_(value)
        copied.add(name)

    def copy_if_exists(name, key):
        # Parameters without a counterpart keep their random initialization.
        if key in source:
            copy(name, source(key))

    # Embeddings
    copy("img_in.weight", permute_img_in(source("img_in.weight"), params["img_in.weight"].data, model.in_dim, model.mask_dim))
    copy("img_in.bias", source("img_in.bias"))
    copy("time_embedding.0.weight", source("time_text_embed.timestep_embedder.linear_1.weight"))
    copy("time_embedding.0.bias", source("time_text_embed.timestep_embedder.linear_1.bias"))
    copy("time_embedding.2.weight", source("time_text_embed.timestep_embedder.linear_2.weight"))
    time_bias = source("time_text_embed.timestep_embedder.linear_2.bias")
    if "time_text_embed.addition_t_embedding.weight" in source:
        time_bias = time_bias + source("time_text_embed.addition_t_embedding.weight")[0]
    copy("time_embedding.2.bias", time_bias)
    copy("text_embedding.0.weight", source("txt_norm.weight"))
    copy("text_embedding.1.weight", source("txt_in.weight"))
    copy("text_embedding.1.bias", source("txt_in.bias"))

    # Blocks: one-to-one, block j <- source block j, the same parameter names.
    if len(model.blocks) != source.num_blocks:
        raise ValueError(f"LayerEditDiT has {len(model.blocks)} blocks, Qwen-Image-Layered has {source.num_blocks}.")
    for name in params:
        if (m := re.match(r"blocks\.(\d+)\.(.+)", name)):
            copy_if_exists(name, f"transformer_blocks.{m.group(1)}.{m.group(2)}")

    # Head
    copy("head.linear.weight", source("norm_out.linear.weight"))
    copy("head.linear.bias", source("norm_out.linear.bias"))
    copy("head.head.weight", permute_head(source("proj_out.weight"), model.out_dim))
    copy("head.head.bias", permute_head(source("proj_out.bias"), model.out_dim))

    # img_in.weight is copied but its mask columns stay random.
    random_init = ["img_in.weight (mask columns)"] + sorted(set(params) - copied)
    dropped = sorted({re.sub(r"^transformer_blocks\.\d+\.", "transformer_blocks.*.", key) for key in source.index if key not in source.used})
    return dropped, random_init


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Built directly in bf16 (the output dtype): random init first, then the available source weights are
    # computed in fp32 and cast on copy. Halves the peak CPU memory.
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    model = LayerEditDiT()
    torch.set_default_dtype(torch.float32)
    model.init_weights()
    source = QwenSource(args.qwen_path)
    with torch.no_grad():
        dropped, random_init = init_from_qwen_layered(model, source)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Dropped source tensors (text stream): {dropped}")
    print(f"Randomly initialized: {random_init}")
    print(f"Total parameters: {num_params / 1e9:.3f}B.")

    state_dict = {name: value.detach().to(dtype=torch.bfloat16, device="cpu").contiguous() for name, value in model.state_dict().items()}
    del model
    save_file(state_dict, args.output_path, metadata={"format": "pt", "model": "LayerEditDiT", "init_from": "Qwen-Image-Layered", "source_blocks": json.dumps(list(range(source.num_blocks)))})
    print(f"Saved: {args.output_path}")
    print(f"Model hash: {hash_model_file(args.output_path)} (registered in diffsynth/configs/model_configs.py -> layer_edit_series)")


if __name__ == "__main__":
    main()
