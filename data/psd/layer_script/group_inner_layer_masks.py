"""Generate group-inner samples with a separate mask for every child layer.

Usage:
    python group_inner_layer_masks.py <layers_grouped_dir> <output_dir>
    python group_inner_layer_masks.py ../layers_grouped ../layers_grouped_inner_output --only 01_dog
    python group_inner_layer_masks.py ../layers_grouped ../layers_grouped_inner_output --workers 3 --max_pixels 2250000

The sample layout matches psd_dataset.py's group_inner mode: full image, merged layers below,
the group's child layers, and merged layers above. Unlike group_inner, each child mask is built
from that child's own alpha instead of sharing the folder mask. Downscaling preserves thin alpha
coverage so a narrow layer cannot disappear from its image or mask.
"""
import argparse
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Set, Tuple

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from layer_folder import is_layer_folder, load_layer_folder  # noqa: E402
from psd_utils import (MASK_EXPAND_RATIO, blur_transparent_rgb, boundary_mask, clean_layer_name,
                       render_on_canvas, stack_canvases)  # noqa: E402

DEFAULT_MAX_PIXELS = 2_250_000
RESIZE_POLICY = "alpha_coverage"
OUTPUT_SUFFIX = ".output.json"
DONE_FILENAME = "done.json"
MERGED_FILENAME = "merged_output.json"
MODE = "group_inner_layer_masks"
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def write_json(path: Path, value) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=1)
    os.replace(temp_path, path)


def save_asset(name: str, render: Callable[[], Image.Image], directory: Path, saved: Set[str]) -> str:
    file_name = f"{name}.png"
    if name not in saved:
        render().save(directory / file_name)
        saved.add(name)
    return file_name


def target_size(image: Image.Image, max_pixels: int):
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image.size
    scale = math.sqrt(max_pixels / pixels)
    return max(1, int(width * scale)), max(1, int(height * scale))


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Resize an image without changing aspect ratio, preserving any covered alpha pixels."""
    size = target_size(image, max_pixels)
    if size == image.size:
        return image
    resized = image.resize(size, resample=Image.Resampling.LANCZOS)
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        coverage = alpha.resize(size, resample=Image.Resampling.BOX)
        coverage = coverage.point(lambda value: 255 if value else 0)
        resized_alpha = ImageChops.lighter(resized.getchannel("A"), coverage)
        if alpha.getbbox() and not resized_alpha.getbbox():
            x0, y0, x1, y1 = alpha.getbbox()
            px = min(size[0] - 1, int(((x0 + x1) / 2) * size[0] / image.width))
            py = min(size[1] - 1, int(((y0 + y1) / 2) * size[1] / image.height))
            resized_alpha.putpixel((px, py), 255)
        resized.putalpha(resized_alpha)
    return resized


def resize_mask_to_max_pixels(mask: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale masks by coverage so a thin white region cannot vanish between sample points."""
    size = target_size(mask, max_pixels)
    if size == mask.size:
        return mask
    source = mask.convert("RGB").getchannel("R")
    source_bbox = source.getbbox()
    coverage = source.resize(size, resample=Image.Resampling.BOX)
    coverage = coverage.point(lambda value: 255 if value else 0)
    if source_bbox and not coverage.getbbox():
        x0, y0, x1, y1 = source_bbox
        px = min(size[0] - 1, int(((x0 + x1) / 2) * size[0] / mask.width))
        py = min(size[1] - 1, int(((y0 + y1) / 2) * size[1] / mask.height))
        coverage.putpixel((px, py), 255)
    return Image.merge("RGBA", (coverage, coverage, coverage, Image.new("L", size, 255)))


def process_model(model_dir: Path, output_dir: Path, relative_dir: Path, blur: bool = True,
                  max_pixels: int = DEFAULT_MAX_PIXELS) -> List[dict]:
    size, groups = load_layer_folder(model_dir)
    if not groups:
        raise ValueError(f"No non-empty groups found in {model_dir}")

    canvases = [render_on_canvas(group, size) for group in groups]
    full = lambda: stack_canvases(canvases)
    output_model = output_dir / relative_dir / model_dir.name
    output_model.mkdir(parents=True, exist_ok=True)
    prefix = (relative_dir / model_dir.name).as_posix()
    saved_images, saved_masks = set(), set()
    entries = []

    for group_index, group in enumerate(groups):
        children = [child for child in group if child.is_visible()]
        if not children:
            continue

        image_assets: List[Tuple[str, Callable[[], Image.Image]]] = [("full", full)]
        mask_assets: List[Tuple[str, Callable[[], Image.Image]]] = [
            ("mask_white", lambda: Image.new("RGBA", size, (255, 255, 255, 255)))
        ]
        if group_index:
            image_assets.append((f"below_{group_index:03d}", lambda i=group_index: stack_canvases(canvases[:i])))
            mask_assets.append(("mask_white", lambda: Image.new("RGBA", size, (255, 255, 255, 255))))

        for child_index, child in enumerate(children):
            canvas = lambda child=child: render_on_canvas(child, size)
            image_assets.append((f"item_{group_index:03d}_child_{child_index:03d}", canvas))
            mask_assets.append((f"item_{group_index:03d}_child_{child_index:03d}_mask",
                                lambda canvas=canvas: boundary_mask(canvas())))

        if group_index < len(groups) - 1:
            image_assets.append((f"above_{group_index:03d}",
                                 lambda i=group_index: stack_canvases(canvases[i + 1:])))
            mask_assets.append(("mask_white", lambda: Image.new("RGBA", size, (255, 255, 255, 255))))

        image_paths = []
        for name, render in image_assets:
            if name not in saved_images:
                image = render()
                if blur:
                    image = blur_transparent_rgb(image)
                image = resize_to_max_pixels(image, max_pixels)
                image.save(output_model / f"{name}.png")
                saved_images.add(name)
            image_paths.append(f"{prefix}/{name}.png")

        mask_paths = []
        for name, render in mask_assets:
            mask_file = save_asset(
                name,
                lambda render=render: resize_mask_to_max_pixels(render(), max_pixels),
                output_model,
                saved_masks,
            )
            mask_paths.append(f"{prefix}/{mask_file}")

        names = [name for name in map(clean_layer_name, (child.name for child in children)) if name]
        if names:
            prompt = f"拆分出{','.join(dict.fromkeys(names))}，共{len(names)}层。"
        else:
            prompt = f"解析一级组 {group.name}，无有效子图层"
        entries.append({
            "image": image_paths,
            "mask": mask_paths,
            "layer_num": len(image_paths) - 1,
            "layer_input_image": image_paths[0],
            "prompt": prompt,
        })

    write_json(output_model / f"{model_dir.name}{OUTPUT_SUFFIX}", entries)
    write_json(output_model / DONE_FILENAME, {
        "mode": MODE,
        "blur": blur,
        "mask_expand_ratio": MASK_EXPAND_RATIO,
        "max_pixels": max_pixels,
        "resize_policy": RESIZE_POLICY,
    })
    return entries


def collect_output_jsons(output_dir: Path) -> List[dict]:
    merged = []
    for path in sorted(output_dir.rglob(f"*{OUTPUT_SUFFIX}")):
        if (path.parent / DONE_FILENAME).exists() and path.name == f"{path.parent.name}{OUTPUT_SUFFIX}":
            with open(path, encoding="utf-8") as file:
                merged.extend(json.load(file))
    return merged


def init_worker() -> None:
    import cv2
    cv2.setNumThreads(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate group-inner samples with per-layer masks")
    parser.add_argument("layers_dir", type=Path, help="Grouped layer folders (output of name_layers.py)")
    parser.add_argument("output_dir", type=Path, help="Output dataset folder")
    parser.add_argument("--only", nargs="+", help="Process these model folders only")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate model folders that are already done")
    parser.add_argument("--no-blur", action="store_true", help="Skip blurring transparent image RGB")
    parser.add_argument("--workers", type=int, default=3, help="Models processed in parallel (default: 3)")
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help=f"Maximum pixels per image, preserving aspect ratio (default: {DEFAULT_MAX_PIXELS})")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace", line_buffering=True)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_pixels < 1:
        parser.error("--max_pixels must be at least 1")
    if not args.layers_dir.is_dir():
        parser.error(f"Input folder does not exist: {args.layers_dir}")

    models = sorted(
        path.parent for path in args.layers_dir.rglob("layers.json")
        if path.parent != args.layers_dir and is_layer_folder(path.parent)
    )
    if args.only:
        selected = set(args.only)
        models = [path for path in models if path.name in selected]
    if not models:
        parser.error(f"No grouped layer folders found in: {args.layers_dir}")

    blur = not args.no_blur
    todo = []
    skipped = 0
    for index, model_dir in enumerate(models, start=1):
        relative_dir = model_dir.parent.relative_to(args.layers_dir)
        output_model = args.output_dir / relative_dir / model_dir.name
        done_path = output_model / DONE_FILENAME
        output_path = output_model / f"{model_dir.name}{OUTPUT_SUFFIX}"
        if not args.overwrite and done_path.exists() and output_path.exists():
            try:
                with open(done_path, encoding="utf-8") as file:
                    done = json.load(file)
                if done == {"mode": MODE, "blur": blur, "mask_expand_ratio": MASK_EXPAND_RATIO,
                            "max_pixels": args.max_pixels, "resize_policy": RESIZE_POLICY}:
                    print(f"[{index}/{len(models)}] {model_dir.name}: skipped")
                    skipped += 1
                    continue
            except (OSError, ValueError):
                pass
        todo.append((index, model_dir, relative_dir))

    failed = []
    if args.workers == 1:
        for index, model_dir, relative_dir in todo:
            try:
                entries = process_model(model_dir, args.output_dir, relative_dir, blur, args.max_pixels)
                print(f"[{index}/{len(models)}] {model_dir.name}: {len(entries)} samples")
            except Exception as error:
                print(f"[{index}/{len(models)}] {model_dir.name}: failed: {error!r}")
                failed.append(model_dir)
    else:
        for name in THREAD_ENV_VARS:
            os.environ[name] = "1"
        with ProcessPoolExecutor(
            args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,
        ) as executor:
            futures = [
                (index, model_dir, executor.submit(
                    process_model, model_dir, args.output_dir, relative_dir, blur, args.max_pixels))
                for index, model_dir, relative_dir in todo
            ]
            for index, model_dir, future in futures:
                try:
                    entries = future.result()
                    print(f"[{index}/{len(models)}] {model_dir.name}: {len(entries)} samples")
                except Exception as error:
                    print(f"[{index}/{len(models)}] {model_dir.name}: failed: {error!r}")
                    failed.append(model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = collect_output_jsons(args.output_dir)
    write_json(args.output_dir / MERGED_FILENAME, merged)
    print(f"Done: {len(models) - skipped - len(failed)} succeeded, {skipped} skipped, {len(failed)} failed, "
          f"{len(merged)} samples in {args.output_dir / MERGED_FILENAME}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
