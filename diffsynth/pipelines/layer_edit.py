import torch
import numpy as np
from einops import rearrange
from PIL import Image
from typing import Union
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.layer_edit_dit import LayerEditDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from .qwen_image import QwenImageUnit_PromptEmbedder


class LayerEditPipeline(BasePipeline):
    """
    Image -> Live2D layered PSD.

    Latent layout: (1, 16, 1 + n, H // 8, W // 8)
        frame 0    : layer_input_image (clean, fused into the latents like image-to-video)
        frame 1..n : the n RGBA layers to be generated
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer

        self.scheduler = FlowMatchScheduler("Wan")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: LayerEditDiT = None
        self.vae: QwenImageVAE = None
        self.tokenizer: Qwen2Tokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            LayerEditUnit_ShapeChecker(),
            LayerEditUnit_NoiseInitializer(),
            LayerEditUnit_PromptEmbedder(),
            LayerEditUnit_LayerInputImageEmbedder(),
            LayerEditUnit_InputLayersEmbedder(),
            LayerEditUnit_MaskEmbedder(),
        ]
        self.model_fn = model_fn_layer_edit
        self.compilable_models = ["dit"]


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = LayerEditPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("layer_edit_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    def encode_rgba_image(self, image: Image.Image, height: int, width: int):
        image = image.convert("RGBA").resize((width, height))
        image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
        latents = self.vae.encode(image)
        return latents.unsqueeze(2)  # (1, C, 1, H, W)


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Condition image (the full illustration to be split into layers)
        layer_input_image: Image.Image = None,
        # Number of layers to generate
        layer_num: int = 4,
        # Segmentation masks (RGB/RGBA/None), aligned with [layer_input_image, layer_1, ..., layer_n]; missing = all white
        mask: list[Image.Image] = None,
        # Layers (optional, for denoising_strength < 1)
        input_image: list[Image.Image] = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        if layer_input_image is not None and (height is None or width is None):
            width, height = layer_input_image.size

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "layer_input_image": layer_input_image,
            "layer_num": layer_num,
            "mask": mask,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Decode (skip the condition frame)
        self.load_models_to_device(["vae"])
        latents = inputs_shared["latents"]
        if "first_frame_latents" in inputs_shared:
            latents = latents[:, :, inputs_shared["first_frame_latents"].shape[2]:]
        layers = []
        for layer_id in range(latents.shape[2]):
            image = self.vae.decode(latents[:, :, layer_id])
            layers.append(self.vae_output_to_image(image))
        self.load_models_to_device([])
        return layers



class LayerEditUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: LayerEditPipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class LayerEditUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "layer_num", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: LayerEditPipeline, height, width, layer_num, seed, rand_device):
        # 1 condition frame + layer_num layers
        noise = pipe.generate_noise((1, 16, 1 + layer_num, height // 8, width // 8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class LayerEditUnit_PromptEmbedder(QwenImageUnit_PromptEmbedder):
    def __init__(self):
        PipelineUnit.__init__(
            self,
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: LayerEditPipeline, prompt) -> dict:
        if pipe.text_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        split_hidden_states = self.encode_prompt(pipe, [prompt])
        context = split_hidden_states[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"context": context}



class LayerEditUnit_LayerInputImageEmbedder(PipelineUnit):
    """
    Encode layer_input_image with the RGBA VAE and fuse it into the first frame of the latents,
    in the same way as Wan2.2-TI2V image-to-video generation.
    """
    def __init__(self):
        super().__init__(
            input_params=("layer_input_image", "noise", "height", "width"),
            output_params=("first_frame_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: LayerEditPipeline, layer_input_image, noise, height, width):
        if layer_input_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        first_frame_latents = pipe.encode_rgba_image(layer_input_image, height, width)
        return {"first_frame_latents": first_frame_latents}



class LayerEditUnit_InputLayersEmbedder(PipelineUnit):
    """
    Build the initial latents: [layer_input_image | noise of n layers].
    During training, `input_image` contains the n target layers and `input_latents` is
    [layer_input_image | n target layers].
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "first_frame_latents", "height", "width"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: LayerEditPipeline, input_image, noise, first_frame_latents, height, width):
        latents = noise.clone()
        if first_frame_latents is not None:
            latents[:, :, 0:1] = first_frame_latents
        if input_image is None:
            return {"latents": latents}
        pipe.load_models_to_device(self.onload_model_names)
        if not isinstance(input_image, list):
            input_image = [input_image]
        layer_latents = torch.concat([pipe.encode_rgba_image(image, height, width) for image in input_image], dim=2)
        if first_frame_latents is not None:
            input_latents = torch.concat([first_frame_latents, layer_latents], dim=2)
        else:
            input_latents = torch.concat([noise[:, :, 0:1], layer_latents], dim=2)
        if input_latents.shape[2] != noise.shape[2]:
            raise ValueError(f"The number of layers ({input_latents.shape[2] - 1}) does not match `layer_num` ({noise.shape[2] - 1}).")
        if pipe.scheduler.training:
            return {"latents": latents, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            if first_frame_latents is not None:
                latents[:, :, 0:1] = first_frame_latents
            return {"latents": latents, "input_latents": input_latents}



class LayerEditUnit_MaskEmbedder(PipelineUnit):
    """
    "mask": list[Image (RGB/RGBA) | None], aligned with the 1 + n frames:
        mask[0] -> layer_input_image, mask[1..n] -> the n layers.
        Missing entries (None, or a list shorter than 1 + n) default to all-white (= 1).
        -> RGB values only (alpha is ignored) -> luminance -> single-channel tensors in [0, 1] (white = 1, black = 0)
        -> downsampled to the latent resolution
        -> replicated to `dit.mask_dim` (4) channels
        -> masks (1, 4, 1 + n, H // 8, W // 8)
    The DiT concatenates masks with the latents on the channel dimension before patchify.
    """
    def __init__(self):
        super().__init__(
            input_params=("mask", "layer_num", "height", "width"),
            output_params=("masks",),
        )

    @staticmethod
    def mask_to_tensor(mask: Image.Image, height, width):
        # convert("RGB") drops the alpha channel without compositing, so only the RGB values are used.
        mask = mask.convert("RGB").convert("L").resize((width, height))
        return torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)

    def process(self, pipe: LayerEditPipeline, mask, layer_num, height, width):
        if mask is None:
            return {}  # The DiT uses all-white masks.
        if not isinstance(mask, list):
            mask = [mask]
        num_frames = 1 + layer_num
        if len(mask) > num_frames:
            raise ValueError(f"Got {len(mask)} masks, but only 1 + {layer_num} frames (layer_input_image + layers).")
        mask = mask + [None] * (num_frames - len(mask))
        masks = torch.stack([
            torch.ones((height, width)) if image is None else self.mask_to_tensor(image, height, width)
            for image in mask
        ])  # (1 + n, H, W)
        masks = torch.nn.functional.interpolate(masks.unsqueeze(1), size=(height // 8, width // 8), mode="area")  # (1 + n, 1, h, w)
        masks = masks.repeat(1, pipe.dit.mask_dim, 1, 1)  # (1 + n, 4, h, w)
        masks = rearrange(masks, "f c h w -> 1 c f h w")
        return {"masks": masks.to(dtype=pipe.torch_dtype, device=pipe.device)}



def parse_rgba_loss_blocks(config: str):
    """
    "0:0.2:0.8,1-2:0.8:0.2" -> [(0, 0, 0.2, 0.8), (1, 2, 0.8, 0.2)]
    Each item is `first_block-last_block:rgb_loss_weight:alpha_loss_weight` (inclusive block ids).
    """
    if config is None or config.strip() == "":
        return []
    groups = []
    for item in config.split(","):
        block_range, rgb_loss_weight, alpha_loss_weight = item.strip().split(":")
        start, end = (block_range.split("-") + [block_range])[:2]
        groups.append((int(start), int(end), float(rgb_loss_weight), float(alpha_loss_weight)))
    groups = sorted(groups)
    for (s0, e0, _, _), (s1, _, _, _) in zip(groups[:-1], groups[1:]):
        if s1 <= e0:
            raise ValueError(f"Overlapping block groups in `{config}`.")
    for start, end, _, _ in groups:
        if start > end:
            raise ValueError(f"Invalid block range {start}-{end} in `{config}`.")
    return groups



def rgba_losses(pipe: LayerEditPipeline, x0_pred: torch.Tensor, x0_target: torch.Tensor):
    """
    Decode the predicted / target latents with the (frozen) RGBA VAE and return (rgb_loss, alpha_loss).
    The RGB loss is weighted by the target alpha, since colors of transparent pixels are meaningless.
    """
    x0_pred = rearrange(x0_pred, "b c f h w -> (b f) c h w").to(pipe.torch_dtype)
    x0_target = rearrange(x0_target, "b c f h w -> (b f) c h w").to(pipe.torch_dtype)
    image_pred = pipe.vae.decode(x0_pred).float().clamp(-1, 1)
    with torch.no_grad():
        image_target = pipe.vae.decode(x0_target).float().clamp(-1, 1)
    alpha_pred = (image_pred[:, 3:4] + 1) / 2
    alpha_target = (image_target[:, 3:4] + 1) / 2
    alpha_loss = torch.nn.functional.mse_loss(alpha_pred, alpha_target)
    rgb_error = (image_pred[:, :3] - image_target[:, :3]).pow(2) * alpha_target
    rgb_loss = rgb_error.sum() / (alpha_target.sum() * 3).clamp_min(1.0)
    return rgb_loss, alpha_loss



def LayerEditRGBALoss(
    pipe: LayerEditPipeline,
    rgba_loss_blocks: list = (),
    **inputs,
):
    """
    Flow-matching loss (global) + block-routed RGB / alpha losses.

    1. The full DiT forward pass produces the final prediction v.
    2. x0 = x_t - sigma * v is decoded by the RGBA VAE, and RGB / alpha losses are computed
       against the decoded ground truth (the condition frame is excluded).
    3. The flow-matching loss updates every parameter as usual. The RGB / alpha losses only update
       the blocks listed in `rgba_loss_blocks`, each group with its own weights, e.g. the default
       cyclic pattern (A, RGB, RGB) * 10:
           A blocks   (0, 3, ..., 27): 0.2 * rgb_loss + 0.8 * alpha_loss
           RGB blocks (the others)   : 0.8 * rgb_loss + 0.2 * alpha_loss
       All other parameters (embeddings, head) receive no gradient from them.

    How the gradient is routed (all in one training step, with a single backward pass):
        - g_rgb / g_alpha = d(L_rgb) / d(L_alpha) w.r.t. the output of every group's last block are
          computed with two torch.autograd.grad calls on the existing graph (one per loss, covering all
          groups at once, so the cost does not grow with the number of groups). The gradient of a group
          is then w_rgb * g_rgb + w_a * g_alpha. This is only a gradient through the rest of the
          network; no loss is computed on intermediate hidden states.
        - The group is re-run on detached inputs (hidden state, text context, timestep modulation,
          RoPE), and the surrogate term sum(y * g) is added to the loss. Its gradient w.r.t.
          the group's parameters equals w_rgb * dL_rgb/dθ + w_a * dL_a/dθ exactly, and it cannot
          reach any other parameter.
    The returned loss value is therefore not the sum of all losses; only its gradient is meaningful.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    num_condition_layers = 0
    if "first_frame_latents" in inputs:
        num_condition_layers = inputs["first_frame_latents"].shape[2]
        inputs["latents"][:, :, :num_condition_layers] = inputs["first_frame_latents"]

    # 1. Full DiT forward pass
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    use_rgba_loss = len(rgba_loss_blocks) > 0
    output = pipe.model_fn(**models, **inputs, timestep=timestep, return_hidden_states=use_rgba_loss)
    noise_pred, hidden_states, block_kwargs = output if use_rgba_loss else (output, None, None)

    training_weight = pipe.scheduler.training_weight(timestep)
    loss = torch.nn.functional.mse_loss(
        noise_pred[:, :, num_condition_layers:].float(),
        training_target[:, :, num_condition_layers:].float(),
    ) * training_weight
    if not use_rgba_loss:
        return loss

    if pipe.vae is None:
        raise ValueError("The RGB / alpha losses require the RGBA VAE. Please add it to `--model_id_with_origin_paths` or `--model_paths`.")
    dit = pipe.dit
    num_blocks = len(dit.blocks)
    for start, end, _, _ in rgba_loss_blocks:
        if end >= num_blocks:
            raise ValueError(f"Block range {start}-{end} is out of range, the DiT only has {num_blocks} blocks.")

    # 2. Decode the predicted x0 and compute the RGB / alpha losses.
    #    The gradients are taken w.r.t. a detached copy of v, so the DiT graph is not touched here.
    noise_pred_layers = noise_pred[:, :, num_condition_layers:].detach().requires_grad_(True)
    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=noise_pred.dtype, device=noise_pred.device)
    x0_pred = inputs["latents"][:, :, num_condition_layers:] - sigma * noise_pred_layers
    x0_target = inputs["input_latents"][:, :, num_condition_layers:]
    rgb_loss, alpha_loss = rgba_losses(pipe, x0_pred, x0_target)
    grad_rgb, = torch.autograd.grad(rgb_loss, noise_pred_layers, retain_graph=True)
    grad_alpha, = torch.autograd.grad(alpha_loss, noise_pred_layers)
    del x0_pred, noise_pred_layers

    # 3. Route the weighted RGB / alpha gradient to each block group only.
    #    The gradient w.r.t. a group output is linear in grad_output, so two backward passes
    #    (one per loss) give the gradients of all groups, whatever their weights.
    group_outputs = [hidden_states[end + 1] for _, end, _, _ in rgba_loss_blocks]
    grad_groups = []
    for grad_layers in (grad_rgb, grad_alpha):
        grad_output = torch.zeros_like(noise_pred)
        grad_output[:, :, num_condition_layers:] = grad_layers * training_weight
        grad_groups.append(torch.autograd.grad(noise_pred, group_outputs, grad_outputs=grad_output, retain_graph=True))
    del group_outputs, grad_output, grad_rgb, grad_alpha
    detached_kwargs = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in block_kwargs.items()}
    surrogate = 0
    for (start, end, rgb_loss_weight, alpha_loss_weight), grad_group_rgb, grad_group_alpha in zip(rgba_loss_blocks, *grad_groups):
        grad_group_output = rgb_loss_weight * grad_group_rgb.float() + alpha_loss_weight * grad_group_alpha.float()
        group_output = dit.forward_blocks(
            hidden_states[start].detach(), start, end + 1, **detached_kwargs,
            use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
            use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
        )
        surrogate = surrogate + (group_output.float() * grad_group_output).sum()

    # The surrogate has no meaningful value, only its gradient. Keep the reported loss = flow-matching loss.
    loss = loss + surrogate - surrogate.detach()
    return loss



def model_fn_layer_edit(
    dit: LayerEditDiT = None,
    latents=None,
    timestep=None,
    context=None,
    masks=None,
    first_frame_latents=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    return_hidden_states=False,
    **kwargs,
):
    num_condition_layers = 0 if first_frame_latents is None else first_frame_latents.shape[2]
    return dit(
        latents, timestep, context,
        masks=masks,
        num_condition_layers=num_condition_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        return_hidden_states=return_hidden_states,
    )
