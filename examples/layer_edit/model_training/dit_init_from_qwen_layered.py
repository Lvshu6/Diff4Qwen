"""
examples/layer_edit/model_training/dit_init_from_qwen_layered.py
Initialize LayerEditDiT from the Qwen-Image-Layered DiT and save bf16 safetensors (same output format as dit_init.py).

Neither model is modified. Every LayerEditDiT tensor is either copied from a Qwen-Image-Layered tensor
(directly, permuted, tiled, averaged or padded), or keeps the random init of `LayerEditDiT.init_weights`.

Source: Qwen-Image-Layered, MMDiT with 60 dual-stream blocks, dim 3072, 24 heads x 128, MLP 12288.
Target: LayerEditDiT, Wan-style single-stream blocks in the cycle (A, B, B): A = self + text attention, B = self only.
The `base` preset (dim 3072, 24 heads x 128) has the same width, so all block weights are reused at full size.

Mapping (source -> target):
    img_in [3072, 64] (C P Q)        -> img_in [dim, 80] (P Q C), 16 latent channels permuted, 4 mask channels zero
    time_text_embed.timestep_embedder -> time_embedding (same sinusoidal embedding, t in [0, 1000], cos first)
    addition_t_embedding[0]          -> folded into time_embedding.2.bias (Qwen-Image-Layered always uses index 0)
    txt_norm + txt_in                -> text_embedding.0 (RMSNorm gain folded in, 1 / typical RMS of the text
                                        features folded in, see --text_rms_cache_dir), text_embedding.2 = identity
    block i (uniformly picked, see --block_map):
        attn.to_q/k/v/to_out.0       -> self_attn.q/k/v/o
        attn.norm_q/k [128]          -> self_attn.norm_q/k [dim] (per-head gain tiled over the heads)
        attn.to_q, add_k/v_proj      -> cross_attn.q/k/v (A blocks: image queries, text keys / values)
        attn.norm_q, norm_added_k    -> cross_attn.norm_q/k (tiled)
        to_out.0                     -> cross_attn.o (--cross_attn_out copy), or zeros (zero: default)
        img_mlp.net.0.proj / net.2   -> ffn.0 / ffn.2 (hidden 12288 -> ffn_dim)
        img_mod.1 = W_i s(t) + b_i    -> time_projection.1 = mean_i(W_i), mean_i(b_i)  (shared by all blocks)
                                        blocks[j].modulation = b_i - mean_i(b_i) + (W_i - mean_i(W_i)) mean_t(s(t))
                                        (exact at the mean timestep feature, the error over t is printed)
    norm_out.linear bias (scale, shift) -> head.modulation (shift, scale)
    proj_out [64, 3072] (C P Q)      -> head.head [64, dim] (P Q C)

Shape mismatches:
    new input features (extra columns)            -> zero, so they have no effect at init
    new outputs that are written to the residual  -> zero (img_in, o, ffn.2, text_embedding.2, modulation)
    new attention heads / MLP hidden units        -> keep the random init (their consumer columns are zero,
                                                     so the output is unchanged but they still get gradients)
    new residual dims (dim > 3072)                -> gates set to 1, so the zero-initialized writers still train
    smaller target (dim < 3072, ffn_dim < 12288)  -> truncation (lossy)

Not representable without changing the code (fine-tuning has to adapt them):
    - RoPE: Qwen uses axes (16, 56, 56) with centered positions, LayerEditDiT uses (44, 42, 42) from 0.
    - Joint text-image attention -> self-attention + separate cross-attention in A blocks only.
    - Per-block timestep projections -> one shared projection (the error is printed).
    - Per-head q/k RMSNorm -> full-width RMSNorm; final AdaLN projection -> Wan head (t added directly).
    - MLP activation x * sigmoid(1.702 x) -> GELU (tanh), both approximate GELU.
"""
import torch, os, re, glob, json, argparse
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from diffsynth.core.loader import hash_model_file
from diffsynth.models.layer_edit_dit import LayerEditDiT, LAYER_EDIT_DIT_PRESETS, sinusoidal_embedding_1d


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize LayerEditDiT weights from Qwen-Image-Layered.")
    parser.add_argument("--preset", type=str, default="base", choices=list(LAYER_EDIT_DIT_PRESETS.keys()), help="Model size. `base` has the same width as Qwen-Image-Layered.")
    parser.add_argument("--qwen_path", type=str, default="models/Qwen/Qwen-Image-Layered/transformer", help="Folder or comma-separated safetensors files of the Qwen-Image-Layered DiT.")
    parser.add_argument("--output_path", type=str, default=None, help="Output .safetensors path.")
    parser.add_argument("--block_map", type=str, default="uniform", choices=["uniform", "first", "last"], help="Which source blocks are used when the depths differ.")
    parser.add_argument("--cross_attn_out", type=str, default="zero", choices=["zero", "copy"], help="zero: A blocks start as pure self-attention blocks (cross_attn.o = 0). copy: cross_attn.o = to_out.0.")
    parser.add_argument("--text_rms_cache_dir", type=str, default=None, help="Optional. Cached data (train.py sft:data_process) used to estimate the typical RMS of the text features.")
    parser.add_argument("--text_rms", type=float, default=None, help="Optional. Typical RMS of the text features, overrides --text_rms_cache_dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of the tensors that are not reused.")
    return parser.parse_args()


class QwenSource:
    def __init__(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if os.path.isdir(path) else path.split(",")
        self.handles, self.index = [], {}
        for file in files:
            handle = safe_open(file, framework="pt", device="cpu")
            self.handles.append(handle)
            for key in handle.keys():
                self.index[key] = handle
        self.num_blocks = 1 + max(int(m.group(1)) for k in self.index if (m := re.match(r"transformer_blocks\.(\d+)\.", k)))

    def __contains__(self, key):
        return key in self.index

    def __call__(self, key):
        return self.index[key].get_tensor(key).float()


def fit_(dst: torch.Tensor, src: torch.Tensor, new_rows="zero", new_row_value=0.0):
    # Copy the overlapping region. Extra columns (new input features) are zero, extra rows are
    # `new_row_value` (new_rows="zero") or keep their current values (new_rows="random").
    if dst.dim() == 2 and dst.shape[1] > src.shape[1]:
        dst[:, src.shape[1]:] = 0
    if dst.shape[0] > src.shape[0] and new_rows == "zero":
        dst[src.shape[0]:] = new_row_value
    region = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    dst[region] = src[region].to(dst.dtype)


class Initializer:
    def __init__(self, model: LayerEditDiT):
        self.params = dict(model.named_parameters())
        self.notes = {}

    def record(self, name, src_shape, dst_shape, note):
        self.notes[name] = note if tuple(src_shape) == tuple(dst_shape) else f"{note}, {tuple(src_shape)} -> {tuple(dst_shape)}"

    def place(self, name, src, note="copy", new_rows="zero"):
        fit_(self.params[name].data, src, new_rows=new_rows)
        self.record(name, src.shape, self.params[name].shape, note)

    def place_chunks(self, name, src, num_chunks, note, new_row_values=None):
        # For tensors made of `num_chunks` stacked dim-sized parts (modulation: shift, scale, gate, ...),
        # fit every part on its own: [n * dim, dim] -> (n, dim, dim), [n * dim] or [1, n, dim] -> (n, dim).
        # new_row_values[c] fills the new dims of part c.
        chunked = lambda t: t.view(num_chunks, -1, t.shape[-1]) if t.dim() == 2 else t.reshape(num_chunks, -1)
        dst_chunks, src_chunks = chunked(self.params[name].data), chunked(src)
        for c in range(num_chunks):
            fit_(dst_chunks[c], src_chunks[c], new_row_value=0.0 if new_row_values is None else new_row_values[c])
        self.record(name, src_chunks.shape, dst_chunks.shape, note)


def select_blocks(num_source, num_target, block_map):
    if block_map == "first":
        return [min(j, num_source - 1) for j in range(num_target)]
    if block_map == "last":
        return [max(num_source - num_target + j, 0) for j in range(num_target)]
    if num_target == 1:
        return [0]
    return [round(j * (num_source - 1) / (num_target - 1)) for j in range(num_target)]


def permute_img_in(weight, in_dim=16, mask_dim=4):
    # Qwen patch features are (C P Q), LayerEditDiT patchify gives (P Q C) with C = in_dim + mask_dim (p = 1 for the layer axis).
    # Mask channels get zero columns, i.e. they do not change the output at init.
    out = torch.zeros(weight.shape[0], 4 * (in_dim + mask_dim), dtype=weight.dtype)
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


def tile_head_norm(weight, dim):
    # Per-head RMSNorm gain [head_dim] -> full-width gain [dim], the same gain for every head.
    return weight.repeat((dim + weight.shape[0] - 1) // weight.shape[0])[:dim]


def timestep_features(source: QwenSource, freq_dim, num_timesteps=50):
    # s(t) = silu(temb(t)) of Qwen-Image-Layered for t in [0, 1000], the input of every img_mod.1.
    # Qwen's TimestepEmbeddings(flip_sin_to_cos, scale=1000) on t / 1000 equals Wan's sinusoidal embedding on t.
    timesteps = torch.linspace(0, 1000, num_timesteps)
    temb = sinusoidal_embedding_1d(freq_dim, timesteps).float()
    temb = F.silu(temb @ source("time_text_embed.timestep_embedder.linear_1.weight").T + source("time_text_embed.timestep_embedder.linear_1.bias"))
    temb = temb @ source("time_text_embed.timestep_embedder.linear_2.weight").T + source("time_text_embed.timestep_embedder.linear_2.bias")
    if "time_text_embed.addition_t_embedding.weight" in source:
        temb = temb + source("time_text_embed.addition_t_embedding.weight")[0]
    return F.silu(temb)


def estimate_text_rms(cache_dir, num_files=64):
    # Median per-token RMS of the cached text features (Qwen applies an RMSNorm before txt_in, LayerEditDiT does not).
    files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.pth"), recursive=True))[:num_files]
    values = []
    for file in files:
        data = torch.load(file, map_location="cpu", weights_only=False)
        for item in data if isinstance(data, (list, tuple)) else [data]:
            if isinstance(item, dict) and isinstance(item.get("context"), torch.Tensor):
                context = item["context"].float().flatten(0, -2)
                values.append(context.pow(2).mean(dim=-1).sqrt())
                break
    if len(values) == 0:
        raise ValueError(f"No cached `context` found in {cache_dir}.")
    return torch.cat(values).median().item()


def init_from_qwen_layered(model: LayerEditDiT, source: QwenSource, block_map="uniform", cross_attn_out="zero", text_rms=1.0):
    init = Initializer(model)
    dim, src_dim = model.dim, source("img_in.weight").shape[0]

    # Embeddings
    init.place("img_in.weight", permute_img_in(source("img_in.weight"), model.in_dim, model.mask_dim), "permuted (C P Q) -> (P Q C), mask columns zero")
    init.place("img_in.bias", source("img_in.bias"))
    init.place("time_embedding.0.weight", source("time_text_embed.timestep_embedder.linear_1.weight"))
    init.place("time_embedding.0.bias", source("time_text_embed.timestep_embedder.linear_1.bias"))
    init.place("time_embedding.2.weight", source("time_text_embed.timestep_embedder.linear_2.weight"))
    time_bias = source("time_text_embed.timestep_embedder.linear_2.bias")
    if "time_text_embed.addition_t_embedding.weight" in source:
        time_bias = time_bias + source("time_text_embed.addition_t_embedding.weight")[0]
    init.place("time_embedding.2.bias", time_bias, "copy + addition_t_embedding[0]")
    txt_in = source("txt_in.weight") * source("txt_norm.weight")[None] / text_rms
    init.place("text_embedding.0.weight", txt_in, f"txt_in * txt_norm gain / {text_rms:.4g}")
    init.place("text_embedding.0.bias", source("txt_in.bias"))
    init.place("text_embedding.2.weight", torch.eye(dim), "identity")
    init.place("text_embedding.2.bias", torch.zeros(dim), "zero")

    # Blocks
    source_ids = select_blocks(source.num_blocks, len(model.blocks), block_map)
    # Qwen: mod_i(t) = W_i s(t) + b_i with s(t) = silu(temb(t)), one W_i per block.
    # LayerEditDiT: mod_j(t) = W s(t) + b + m_j with one shared W. Least squares over the blocks gives
    # W = mean_i(W_i); m_j absorbs b_i and the part of (W_i - W) s(t) at the mean s(t), so only the
    # timestep-dependent part of the per-block difference is lost (the error is printed).
    features = timestep_features(source, model.freq_dim)
    features_mean = features.mean(dim=0)
    mod_weight = sum(source(f"transformer_blocks.{i}.img_mod.1.weight") for i in source_ids) / len(source_ids)
    mod_bias = sum(source(f"transformer_blocks.{i}.img_mod.1.bias") for i in source_ids) / len(source_ids)
    init.place_chunks("time_projection.1.weight", mod_weight, 6, "mean of img_mod.1 over the used blocks")
    init.place_chunks("time_projection.1.bias", mod_bias, 6, "mean of img_mod.1 over the used blocks")
    mod_errors = []
    # New residual dims: shift = scale = 0, gate = 1 (the writers are zero, a gate of 1 keeps them trainable).
    gate_one = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

    for j, (block, i) in enumerate(zip(model.blocks, source_ids)):
        s, d = f"transformer_blocks.{i}.", f"blocks.{j}."
        for target, name in (("q", "to_q"), ("k", "to_k"), ("v", "to_v")):
            init.place(d + f"self_attn.{target}.weight", source(s + f"attn.{name}.weight"), new_rows="random")
            init.place(d + f"self_attn.{target}.bias", source(s + f"attn.{name}.bias"), new_rows="random")
        init.place(d + "self_attn.o.weight", source(s + "attn.to_out.0.weight"))
        init.place(d + "self_attn.o.bias", source(s + "attn.to_out.0.bias"))
        init.place(d + "self_attn.norm_q.weight", tile_head_norm(source(s + "attn.norm_q.weight"), dim), "per-head gain tiled")
        init.place(d + "self_attn.norm_k.weight", tile_head_norm(source(s + "attn.norm_k.weight"), dim), "per-head gain tiled")
        if block.has_text_attn:
            for target, name in (("q", "to_q"), ("k", "add_k_proj"), ("v", "add_v_proj")):
                init.place(d + f"cross_attn.{target}.weight", source(s + f"attn.{name}.weight"), new_rows="random")
                init.place(d + f"cross_attn.{target}.bias", source(s + f"attn.{name}.bias"), new_rows="random")
            init.place(d + "cross_attn.norm_q.weight", tile_head_norm(source(s + "attn.norm_q.weight"), dim), "per-head gain tiled")
            init.place(d + "cross_attn.norm_k.weight", tile_head_norm(source(s + "attn.norm_added_k.weight"), dim), "per-head gain tiled")
            init.place(d + "norm3.weight", torch.ones(dim), "LayerNorm identity (no source)")
            init.place(d + "norm3.bias", torch.zeros(dim), "LayerNorm identity (no source)")
            if cross_attn_out == "copy":
                init.place(d + "cross_attn.o.weight", source(s + "attn.to_out.0.weight"))
                init.place(d + "cross_attn.o.bias", source(s + "attn.to_out.0.bias"))
            else:
                init.place(d + "cross_attn.o.weight", torch.zeros(dim, dim), "zero")
                init.place(d + "cross_attn.o.bias", torch.zeros(dim), "zero")
        init.place(d + "ffn.0.weight", source(s + "img_mlp.net.0.proj.weight"), new_rows="random")
        init.place(d + "ffn.0.bias", source(s + "img_mlp.net.0.proj.bias"), new_rows="random")
        init.place(d + "ffn.2.weight", source(s + "img_mlp.net.2.weight"))
        init.place(d + "ffn.2.bias", source(s + "img_mlp.net.2.bias"))
        weight, bias = source(s + "img_mod.1.weight"), source(s + "img_mod.1.bias")
        init.place_chunks(d + "modulation", bias - mod_bias + (weight - mod_weight) @ features_mean, 6, "img_mod.1 - shared part at the mean timestep", new_row_values=gate_one)
        target_mod, shared_mod = features @ weight.T + bias, features @ mod_weight.T + bias + (weight - mod_weight) @ features_mean
        mod_errors.append(((shared_mod - target_mod).norm(dim=-1) / target_mod.norm(dim=-1)).mean().item())

    # Head. Qwen AdaLayerNorm(single) chunks (scale, shift), LayerEditDiT Head chunks (shift, scale).
    norm_out_bias = source("norm_out.linear.bias")
    init.place_chunks("head.modulation", torch.cat([norm_out_bias[src_dim:], norm_out_bias[:src_dim]]), 2, "norm_out.linear.bias (shift, scale)")
    init.place("head.head.weight", permute_head(source("proj_out.weight"), model.out_dim), "permuted (C P Q) -> (P Q C)")
    init.place("head.head.bias", permute_head(source("proj_out.bias"), model.out_dim), "permuted (C P Q) -> (P Q C)")
    return init.notes, source_ids, mod_errors


def main():
    args = parse_args()
    output_path = args.output_path or f"models/LayerEdit/layer_edit_dit_{args.preset}_from_qwen_layered.safetensors"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    text_rms = args.text_rms
    if text_rms is None and args.text_rms_cache_dir is not None:
        text_rms = estimate_text_rms(args.text_rms_cache_dir)
        print(f"Estimated text feature RMS: {text_rms:.4g}")
    if text_rms is None:
        text_rms = 1.0
        print("Warning: text feature RMS unknown, using 1.0. Qwen normalizes the text features before txt_in, LayerEditDiT does not; pass --text_rms_cache_dir.")

    config = LAYER_EDIT_DIT_PRESETS[args.preset]
    print(f"Initializing LayerEditDiT ({args.preset}) from Qwen-Image-Layered: {config}")
    model = LayerEditDiT(**config)
    model.init_weights()
    source = QwenSource(args.qwen_path)
    if model.dim != 3072:
        print(f"Warning: dim {model.dim} != 3072 (Qwen-Image-Layered). Weights are {'padded' if model.dim > 3072 else 'truncated'}, the LayerNorm / RMSNorm statistics change. The `base` preset matches the width exactly.")
    with torch.no_grad():
        notes, source_ids, mod_errors = init_from_qwen_layered(model, source, args.block_map, args.cross_attn_out, text_rms)

    num_params = sum(p.numel() for p in model.parameters())
    reused = sum(p.numel() for name, p in model.named_parameters() if name in notes and notes[name] != "zero" and notes[name] != "identity")
    exact = sum(p.numel() for name, p in model.named_parameters() if name in notes and "->" not in notes[name] and notes[name] not in ("zero", "identity"))
    print(f"Source blocks used: {source_ids}")
    print(f"Relative error of the block modulation (shift / scale / gate) caused by the shared time_projection, over t in [0, 1000]: "
          f"mean {sum(mod_errors) / len(mod_errors):.1%}, max {max(mod_errors):.1%}")
    print(f"Total parameters: {num_params / 1e9:.3f}B. From Qwen-Image-Layered: {reused / num_params:.1%} ({exact / num_params:.1%} with matching shapes).")
    for name in [name for name in dict(model.named_parameters()) if not name.startswith("blocks.") or name.startswith("blocks.0.") or name.startswith("blocks.1.")]:
        print(f"  {name}: {notes.get(name, 'random init')}")
    random_names = [name for name in dict(model.named_parameters()) if name not in notes]
    if len(random_names) > 0:
        print(f"Randomly initialized tensors: {random_names}")

    state_dict = {name: value.detach().to(dtype=torch.bfloat16, device="cpu").contiguous() for name, value in model.state_dict().items()}
    del model
    save_file(state_dict, output_path, metadata={"format": "pt", "model": "LayerEditDiT", "preset": args.preset, "config": json.dumps(config), "init_from": "Qwen-Image-Layered", "source_blocks": json.dumps(source_ids)})
    print(f"Saved: {output_path}")
    print(f"Model hash: {hash_model_file(output_path)} (registered in diffsynth/configs/model_configs.py -> layer_edit_series)")


if __name__ == "__main__":
    main()
