from diffsynth.pipelines.layer_edit import LayerEditPipeline, ModelConfig
from diffsynth.core.data.operators import ImageCropAndResize
from PIL import Image
import torch, json, os, argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dit_path", type=str, default="models/train/LayerEdit_full/epoch-0.safetensors")
parser.add_argument("--dataset_base_path", type=str, default="data/combine")
parser.add_argument("--metadata_path", type=str, default="data/combine/test.json")
parser.add_argument("--output_path", type=str, default="models/train/LayerEdit_full/inference/epoch-0")
parser.add_argument("--max_pixels", type=int, default=262144)
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--cfg_scale", type=float, default=4.0)
parser.add_argument("--seed", type=int, default=0)
# Masks are stored as `layer_split_image` in the metadata. Training reads `mask`, so the DiT was trained with all-white masks.
parser.add_argument("--use_mask", default=False, action="store_true")
parser.add_argument("--onload_dtype", type=str, default="float8_e4m3fn", choices=["float8_e4m3fn", "bfloat16"])
args = parser.parse_args()
ONLOAD_DTYPE = getattr(torch, args.onload_dtype)


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": ONLOAD_DTYPE,
    "onload_device": "cpu",
    "preparing_dtype": ONLOAD_DTYPE,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LayerEditPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=args.dit_path, **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

# Same resolution rule as training (max_pixels, divisible by 16)
resize = ImageCropAndResize(None, None, args.max_pixels, 16, 16)
load_image = lambda path: resize(Image.open(os.path.join(args.dataset_base_path, path)).convert("RGBA"))

with open(args.metadata_path) as f:
    metadata = json.load(f)
for sample_id, data in enumerate(metadata):
    layer_input_image = load_image(data["layer_input_image"])
    width, height = layer_input_image.size
    mask = [load_image(path) for path in data["layer_split_image"]] if args.use_mask and data.get("layer_split_image") else None
    layers = pipe(
        data["prompt"],
        seed=args.seed, num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale,
        height=height, width=width,
        layer_input_image=layer_input_image, layer_num=data["layer_num"], mask=mask,
    )
    output_dir = os.path.join(args.output_path, f"{sample_id:03d}")
    os.makedirs(output_dir, exist_ok=True)
    layer_input_image.save(os.path.join(output_dir, "input.png"))
    for layer_id, layer in enumerate(layers):
        layer.save(os.path.join(output_dir, f"layer_{layer_id + 1}.png"))
        # Ground truth layers, for comparison
        layer_gt_path = data["image"][layer_id + 1]
        load_image(layer_gt_path).save(os.path.join(output_dir, f"layer_{layer_id + 1}_gt.png"))
