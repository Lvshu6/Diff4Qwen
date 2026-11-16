"""Optimized group-inner dataset generation with per-layer masks.

Usage:
    python group_inner_layer_masks_optimized.py <layers_grouped_dir> <output_dir>
    python group_inner_layer_masks_optimized.py ../layers_grouped ../layers_grouped_inner_output --workers 3 --max_pixels 2250000

Layers are rasterized onto the capped canvas before compositing. Group composites and their
below/above stacks are therefore built at output resolution instead of repeatedly processing
the original full-size canvases.
"""
import argparse
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Set, Tuple

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from layer_folder import is_layer_folder, load_layer_folder  # noqa: E402
from psd_utils import MASK_EXPAND_RATIO, blur_transparent_rgb, boundary_mask, clean_layer_name  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MAX_PIXELS = 2_250_000
MODE = "group_inner_layer_masks_optimized"
RESIZE_POLICY = "alpha_coverage"
OUTPUT_SUFFIX = ".output.json"
DONE_FILENAME = "done.json"
MERGED_FILENAME = "merged_output.json"
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def write_json(path: Path, value) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=1)
    os.replace(temp_path, path)


def output_size(canvas_size: Tuple[int, int], max_pixels: int) -> Tuple[int, int]:
    width, height = canvas_size
    pixels = width * height
    if max_pixels < 1:
        raise ValueError("max_pixels must be at least 1")
    if pixels <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / pixels)
    return max(1, int(width * scale)), max(1, int(height * scale))


def resize_rgba(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize with Lanczos RGB and preserve any non-zero alpha coverage with BOX sampling."""
    image = image.convert("RGBA")
    if image.size == size:
        return image
    resized = image.resize(size, Image.Resampling.LANCZOS)
    alpha = image.getchannel("A")
    coverage = alpha.resize(size, Image.Resampling.BOX).point(lambda value: 255 if value else 0)
    resized_alpha = ImageChops.lighter(resized.getchannel("A"), coverage)
    if alpha.getbbox() and not resized_alpha.getbbox():
        x0, y0, x1, y1 = alpha.getbbox()
        px = min(size[0] - 1, int(((x0 + x1) / 2) * size[0] / image.width))
        py = min(size[1] - 1, int(((y0 + y1) / 2) * size[1] / image.height))
        resized_alpha.putpixel((px, py), 255)
    resized.putalpha(resized_alpha)
    return resized


def resized_layer_content(child, canvas_size: Tuple[int, int], target_size: Tuple[int, int]):
    """(resized cropped layer, target offset), retaining even one-pixel details."""
    canvas_width, canvas_height = canvas_size
    target_width, target_height = target_size
    left, top, right, bottom = child.bbox
    sx, sy = target_width / canvas_width, target_height / canvas_height
    x0 = max(0, min(target_width - 1, round(left * sx)))
    y0 = max(0, min(target_height - 1, round(top * sy)))
    x1 = max(x0 + 1, min(target_width, round(right * sx)))
    y1 = max(y0 + 1, min(target_height, round(bottom * sy)))
    content = resize_rgba(child.composite(), (x1 - x0, y1 - y0))
    return content, (x0, y0)


def save_image(name: str, image: Image.Image, directory: Path, saved: Set[str], blur: bool) -> str:
    file_name = f"{name}.png"
    if name not in saved:
        (blur_transparent_rgb(image) if blur else image).save(directory / file_name)
        saved.add(name)
    return file_name


def save_mask(name: str, image: Image.Image, directory: Path, saved: Set[str]) -> str:
    file_name = f"{name}.png"
    if name not in saved:
        image.save(directory / file_name)
        saved.add(name)
    return file_name


def process_model(model_dir: Path, output_dir: Path, relative_dir: Path, blur: bool,
                  max_pixels: int) -> List[dict]:
    canvas_size, groups = load_layer_folder(model_dir)
    if not groups:
        raise ValueError(f"No non-empty groups found in {model_dir}")
    target_size = output_size(canvas_size, max_pixels)

    output_model = output_dir / relative_dir / model_dir.name
    output_model.mkdir(parents=True, exist_ok=True)
    prefix = (relative_dir / model_dir.name).as_posix()
    saved_images, saved_masks = set(), set()
    save_mask("mask_white", Image.new("RGBA", target_size, (255, 255, 255, 255)),
              output_model, saved_masks)

    transparent = Image.new("RGBA", target_size, (0, 0, 0, 0))
    group_canvases = []
    children_by_group = []
    for group_index, group in enumerate(groups):
        children = [child for child in group if child.is_visible()]
        children_by_group.append(children)
        composite = transparent.copy()
        for child_index, child in enumerate(children):
            content, offset = resized_layer_content(child, canvas_size, target_size)
            composite.alpha_composite(content, offset)
            child_canvas = transparent.copy()
            child_canvas.alpha_composite(content, offset)
            image_name = f"item_{group_index:03d}_child_{child_index:03d}"
            save_image(image_name, child_canvas, output_model, saved_images, blur)
            save_mask(f"{image_name}_mask", boundary_mask(child_canvas), output_model, saved_masks)
        group_canvases.append(composite)

    full = transparent.copy()
    for canvas in group_canvases:
        full = Image.alpha_composite(full, canvas)
    save_image("full", full, output_model, saved_images, blur)

    below = transparent.copy()
    for index, canvas in enumerate(group_canvases):
        if index:
            save_image(f"below_{index:03d}", below, output_model, saved_images, blur)
        below = Image.alpha_composite(below, canvas)

    above = transparent.copy()
    for index in range(len(group_canvases) - 1, -1, -1):
        if index < len(group_canvases) - 1:
            save_image(f"above_{index:03d}", above, output_model, saved_images, blur)
        above = Image.alpha_composite(group_canvases[index], above)

    entries = []
    for group_index, group in enumerate(groups):
        children = children_by_group[group_index]
        if not children:
            continue

        image_paths = [f"{prefix}/full.png"]
        mask_paths = [f"{prefix}/mask_white.png"]
        if group_index:
            image_paths.append(f"{prefix}/below_{group_index:03d}.png")
            mask_paths.append(f"{prefix}/mask_white.png")

        for child_index, _ in enumerate(children):
            image_name = f"item_{group_index:03d}_child_{child_index:03d}"
            image_paths.append(f"{prefix}/{image_name}.png")

            mask_name = f"{image_name}_mask"
            mask_paths.append(f"{prefix}/{mask_name}.png")

        if group_index < len(groups) - 1:
            image_paths.append(f"{prefix}/above_{group_index:03d}.png")
            mask_paths.append(f"{prefix}/mask_white.png")

        names = [name for name in map(clean_layer_name, (child.name for child in children)) if name]
        prompt = (f"拆分出{','.join(dict.fromkeys(names))}，共{len(names)}层。" if names
                  else f"解析一级组 {group.name}，无有效子图层")
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
    parser = argparse.ArgumentParser(description="Optimized group-inner samples with per-layer masks")
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
    todo, skipped = [], 0
    expected_done = {
        "mode": MODE,
        "blur": blur,
        "mask_expand_ratio": MASK_EXPAND_RATIO,
        "max_pixels": args.max_pixels,
        "resize_policy": RESIZE_POLICY,
    }
    for index, model_dir in enumerate(models, start=1):
        relative_dir = model_dir.parent.relative_to(args.layers_dir)
        output_model = args.output_dir / relative_dir / model_dir.name
        done_path = output_model / DONE_FILENAME
        output_path = output_model / f"{model_dir.name}{OUTPUT_SUFFIX}"
        if not args.overwrite and done_path.exists() and output_path.exists():
            try:
                with open(done_path, encoding="utf-8") as file:
                    if json.load(file) == expected_done:
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
