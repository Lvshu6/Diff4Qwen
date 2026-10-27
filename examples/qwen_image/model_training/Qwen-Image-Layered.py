from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth import load_state_dict
from PIL import Image
import torch
from modelscope import dataset_snapshot_download


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        # ModelConfig(path=["models/train/Qwen-Image-Layered_distill/epoch-2.safetensors"]),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
# state_dict = load_state_dict("models/train/Qwen-Image-Layered_distill/epoch-0.safetensors")
# pipe.dit.load_state_dict(state_dict)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="layer/image.png"
)
prompt = "The image features a 3D animated character that appears to be a young boy. He is holding a baseball bat with both hands, ready to swing. The character is wearing a white baseball uniform with blue accents, including blue gloves and blue shorts. He has a black cap on his head and is smiling. The background is a solid green color. Above the character, there is a bold text that reads \"WE NEED\" in white capital letters. Below the character, there are three bullet points, each stating \"Add accepted donation here\" in white text. The overall style of the image suggests it might be used for a fundraising campaign or a call for donations."
  
input_image = Image.open("data/example_image_dataset/layer/image.png").convert("RGBA").resize((864, 480))
input_image = Image.open("/root/yxd/Diff4Qwen/data/parquet_data/export/3D-00000-of-00002/000004/image.png").convert("RGBA").resize((864, 480))

images = pipe(
    prompt, seed=0,
    # height=480, width=864,
    layer_input_image=input_image, layer_num=6,
)
for i, image in enumerate(images):
    if i == 0: continue # The first image is the input image.
    image.save(f"image_{i}.png")
