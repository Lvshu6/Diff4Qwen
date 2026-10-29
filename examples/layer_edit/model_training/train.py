"""
examples/layer_edit/model_training/train.py
Differentiated loss training for different DiT blocks: global flow-match loss stays unchanged.
Extra constraints: group-A blocks use RGB loss weight 0.2, Alpha loss weight 0.8;
group-B blocks use RGB loss weight 0.8, Alpha loss weight 0.2.
Supports differentiated training in one training iteration.
Use cyclic pattern A,B,B repeated for the 3*12=36-layer DiT.
Within each triplet:
- 1st block (A): self-attention + text-cross-attention
  (segmentation masks, aligned with [layer_input_image, layer_1, ..., layer_n], enter the DiT as 4 extra
   input channels of the latents, see LayerEditDiT)
- 2nd & 3rd blocks (B): self-attention only
Set this cyclic pattern as the default loss configuration.
"""
import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import *
from diffsynth.pipelines.layer_edit import LayerEditPipeline, LayerEditRGBALoss, parse_rgba_loss_blocks, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cyclic_rgba_loss_blocks(num_blocks=36, cycle_length=3, attn_block_weights=(0.2, 0.8), rgb_block_weights=(0.8, 0.2)):
    # In every cycle, the 1st block (A) gets `attn_block_weights` and the remaining blocks (B) get `rgb_block_weights`.
    # Format: "first-last:rgb_loss_weight:alpha_loss_weight", e.g. "0:0.2:0.8,1-2:0.8:0.2,3:0.2:0.8,...".
    groups = []
    for start in range(0, num_blocks, cycle_length):
        end = min(start + cycle_length, num_blocks) - 1
        groups.append(f"{start}:{attn_block_weights[0]}:{attn_block_weights[1]}")
        if end > start:
            groups.append(f"{start + 1}-{end}:{rgb_block_weights[0]}:{rgb_block_weights[1]}")
    return ",".join(groups)


# RGB / alpha losses routed to block groups, cyclic pattern (A, B, B) * 12 for the 36-block DiT:
#   A (blocks 0, 3, ..., 33; self + text attention): RGB 0.2, alpha 0.8
#   B (the other blocks; self-attention only)      : RGB 0.8, alpha 0.2
# `--rgba_loss_blocks` defaults to None: the same pattern is then built from the loaded DiT (num_layers, cycle_length).
DEFAULT_RGBA_LOSS_BLOCKS = cyclic_rgba_loss_blocks()


def check_rgba_loss_blocks(dit, rgba_loss_blocks):
    # Warn if a block group mixes A and RGB blocks, or if A / RGB blocks are not weighted consistently,
    # e.g. a custom --rgba_loss_blocks that does not follow the (A, B, B) cycle of the DiT.
    weights = {True: set(), False: set()}
    for start, end, rgb_loss_weight, alpha_loss_weight in rgba_loss_blocks:
        for block in dit.blocks[start:end + 1]:
            weights[block.has_text_attn].add((rgb_loss_weight, alpha_loss_weight))
    if len(weights[True]) > 1 or len(weights[False]) > 1 or (weights[True] & weights[False]):
        attn_blocks = [i for i, block in enumerate(dit.blocks) if block.has_text_attn]
        warnings.warn(f"`rgba_loss_blocks` does not match the DiT layout (A blocks: {attn_blocks}). Weights on A blocks: {weights[True]}, on B blocks: {weights[False]}.")


class LayerEditTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        rgba_loss_blocks=None,
        loss_type="rgba",
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = LayerEditPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        # loss_type:
        #   flow_match: the common flow-matching loss (FlowMatchSFTLoss, same as examples/wanvideo/model_training/train.py)
        #   rgba      : flow-matching loss + block-routed RGB / alpha losses (LayerEditRGBALoss)
        if loss_type not in ("flow_match", "rgba"):
            raise ValueError(f"Unknown loss_type `{loss_type}`, expected `flow_match` or `rgba`.")
        self.loss_type = loss_type
        if rgba_loss_blocks is None:
            rgba_loss_blocks = DEFAULT_RGBA_LOSS_BLOCKS if self.pipe.dit is None else cyclic_rgba_loss_blocks(len(self.pipe.dit.blocks), self.pipe.dit.cycle_length)
        self.rgba_loss_blocks = parse_rgba_loss_blocks(rgba_loss_blocks) if loss_type == "rgba" else []
        print(f"Loss type: {loss_type}. RGB / alpha loss block groups (first, last, rgb_weight, alpha_weight): {self.rgba_loss_blocks}")
        if self.pipe.dit is not None and loss_type == "rgba":
            check_rgba_loss_blocks(self.pipe.dit, self.rgba_loss_blocks)
        sft_loss = {
            "flow_match": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "rgba": lambda pipe, inputs_shared, inputs_posi, inputs_nega: LayerEditRGBALoss(pipe, rgba_loss_blocks=self.rgba_loss_blocks, **inputs_shared, **inputs_posi),
        }[loss_type]
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": sft_loss,
            "sft:train": sft_loss,
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_layers(self, data):
        # data["image"]: [composite, layer_1, ..., layer_n] or [layer_1, ..., layer_n]
        # data["layer_input_image"]: composite image (condition)
        layers = data["image"] if isinstance(data["image"], list) else [data["image"]]
        layer_num = int(data["layer_num"]) if data.get("layer_num") is not None else len(layers)
        if len(layers) == layer_num + 1:
            # The first image is the composite, which is already used as the condition frame.
            layers = layers[1:]
        if len(layers) != layer_num:
            raise ValueError(f"Got {len(layers)} layers, but layer_num = {layer_num}.")
        layer_input_image = data.get("layer_input_image")
        if layer_input_image is None:
            layer_input_image = data["image"][0]
        return layers, layer_num, layer_input_image

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        layers, layer_num, layer_input_image = self.parse_layers(data)
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": layers,
            "layer_input_image": layer_input_image,
            "layer_num": layer_num,
            "mask": data.get("mask"),
            "height": layer_input_image.size[1],
            "width": layer_input_image.size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        for extra_input in self.extra_inputs:
            if extra_input not in ("layer_num", "layer_input_image", "mask"):
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def layer_edit_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--loss_type", type=str, default="rgba", choices=["flow_match", "rgba"], help="flow_match: the common flow-matching loss only. rgba: flow-matching loss + block-routed RGB / alpha losses (see --rgba_loss_blocks).")
    parser.add_argument("--rgba_loss_blocks", type=str, default=None, help="RGB / alpha losses (VAE-decoded prediction vs. GT) and the DiT blocks they update, as comma-separated `first-last:rgb_weight:alpha_weight`. Default: cyclic (A, B, B) over all DiT blocks (* 12 for 36 blocks), A blocks 0.2:0.8, B blocks 0.8:0.2. Other parameters only receive the flow-matching gradient. Empty string disables them.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = layer_edit_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    rgba_image_operator = ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)
    # Mask entries may be null (-> all white).
    mask_operator = RouteByType(operator_map=[(str, rgba_image_operator), (type(None), lambda data: None)])
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        special_operator_map={
            # Condition image: the full illustration (RGBA)
            "layer_input_image": rgba_image_operator,
            # Segmentation masks (RGB/RGBA/null), aligned with [layer_input_image, layer_1, ..., layer_n], concatenated to the latents as input channels
            "mask": RouteByType(operator_map=[
                (str, mask_operator),
                (list, SequencialProcess(mask_operator)),
            ]),
            # Target layers (RGBA)
            "image": RouteByType(operator_map=[
                (str, rgba_image_operator),
                (list, SequencialProcess(rgba_image_operator)),
            ]),
        }
    )
    model = LayerEditTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        rgba_loss_blocks=args.rgba_loss_blocks,
        loss_type=args.loss_type,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
