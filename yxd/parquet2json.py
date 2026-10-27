import os
import io
import json
import glob
import re
from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
INPUT_DIR = "/root/yxd/Diff4Qwen/data/parquet_data/data"
OUTPUT_DIR = "/root/yxd/Diff4Qwen/data/parquet_data/export"
MAX_LAYERS = 50  # 注意：虽然这里设为50，但下面的逻辑会限制有效处理的层数上限为30用于判断是否跳过

LAYER_MODE = "paste_on_canvas"     # 或 "resize_only"
ALSO_SAVE_CROPPED = False
USE_REL_PATH = True

# 是否在顶层额外汇总一个 all_data.json
SAVE_GLOBAL_JSON = True

# 【新增配置】最大允许的 layer_num，超过此值将跳过该样本
MAX_ALLOWED_LAYER_NUM = 30


# ============================================================
# 工具
# ============================================================
def decode_image(obj) -> Image.Image:
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj.convert("RGBA")
    if isinstance(obj, dict):
        b = obj.get("bytes")
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGBA")
        p = obj.get("path")
        if p and os.path.exists(p):
            return Image.open(p).convert("RGBA")
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(obj))).convert("RGBA")
    return None


def normalize_box(box, W, H):
    if box is None:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
    except Exception:
        return None
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return None
    if x2 <= 1 and y2 <= 1 and x1 <= 1 and y1 <= 1 and (W > 1 and H > 1):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def resize_layer_to_box(layer_pil: Image.Image, box_wh):
    bw, bh = box_wh
    if layer_pil.size == (bw, bh):
        return layer_pil
    return layer_pil.resize((bw, bh), resample=Image.LANCZOS)


def safe_name(name: str) -> str:
    """把 parquet 文件名清洗成可做文件夹名的字符串"""
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r"[^\w\-.]+", "_", name).strip("_")
    return name or "unnamed"


# ============================================================
# 处理单个 parquet 文件
# ============================================================
def process_one_parquet(pq_path: str, out_root: str):
    """
    处理一个 parquet，返回该文件的 records 列表。
    样本落在 out_root/{sample_id}/ 下。
    该文件的 JSON 落在 out_root/data.json。
    """
    base = safe_name(pq_path)
    file_out_dir = os.path.join(out_root, base)
    os.makedirs(file_out_dir, exist_ok=True)

    print(f"\n>>> Reading {pq_path}")
    df = pq.read_table(pq_path).to_pandas()
    print(f"  rows: {len(df)}")

    records = []

    for local_idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=base)
    ):
        sample_id = f"{local_idx:06d}"
        
        # 【修改点 1】：提前获取 layer_count 并检查是否超过阈值
        raw_layer_count = int(row.get("layer_count") or 0)
        
        # 如果 layer_num 超过 MAX_ALLOWED_LAYER_NUM，直接跳过该行
        if raw_layer_count > MAX_ALLOWED_LAYER_NUM:
            continue
            
        sample_dir = os.path.join(file_out_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        image_names = []

        # ---------- 主图 whole_image ----------
        main_pil = decode_image(row.get("whole_image"))
        if main_pil is None:
            main_pil = decode_image(row.get("base_image"))
        if main_pil is None:
            print(f"[warn] {base}/{sample_id}: 无主图，跳过")
            continue

        W, H = main_pil.size
        main_pil.save(os.path.join(sample_dir, "image.png"))
        image_names.append("image.png")

        # ---------- 图层 ----------
        # 使用经过检查的 layer_count，同时受 MAX_LAYERS 限制
        layer_count = min(raw_layer_count, MAX_LAYERS)

        if ALSO_SAVE_CROPPED:
            crop_dir = os.path.join(sample_dir, "layers_crop")
            os.makedirs(crop_dir, exist_ok=True)

        for i in range(layer_count):
            key_img = f"layer_{i:02d}"
            key_box = f"layer_{i:02d}_box"

            layer_pil = decode_image(row.get(key_img))
            if layer_pil is None:
                continue

            box = normalize_box(row.get(key_box), W, H)

            if box is not None:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                layer_resized = resize_layer_to_box(layer_pil, (bw, bh))

                if LAYER_MODE == "paste_on_canvas":
                    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    canvas.paste(layer_resized, (x1, y1), layer_resized)
                    out_pil = canvas
                else:
                    out_pil = layer_resized

                if ALSO_SAVE_CROPPED:
                    layer_resized.save(
                        os.path.join(crop_dir, f"layer_{i:02d}.png")
                    )
            else:
                if layer_pil.size != (W, H):
                    layer_pil = layer_pil.resize((W, H),
                                                 resample=Image.LANCZOS)
                out_pil = layer_pil

            name = f"image_{i + 1}.png"
            out_pil.save(os.path.join(sample_dir, name))
            image_names.append(name)

        # ---------- prompt ----------
        prompt = str(row.get("whole_caption") or "")

        # ---------- JSON 记录 ----------
        # 相对路径以 OUTPUT_DIR 为根：<base>/<sample_id>/<file>
        if USE_REL_PATH:
            image_rel = [os.path.join(base, sample_id, n) for n in image_names]
            layer_input_rel = os.path.join(base, sample_id, image_names[0])
        else:
            image_rel = image_names
            layer_input_rel = image_names[0]

        records.append({
            "image": image_rel,
            "layer_num": layer_count,
            "layer_input_image": layer_input_rel,
            "prompt": prompt,
        })

    # ---------- 写该 parquet 的 JSON ----------
    json_path = os.path.join(file_out_dir, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {len(records)} samples -> {json_path}")

    return records


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parquet_files = sorted(glob.glob(
        os.path.join(INPUT_DIR, "**", "*.parquet"), recursive=True
    ))
    print(f"Found {len(parquet_files)} parquet files.")
    if not parquet_files:
        return

    all_records = []
    for pq_path in parquet_files:
        recs = process_one_parquet(pq_path, OUTPUT_DIR)
        all_records.extend(recs)

    # ---------- 顶层汇总 JSON（可选） ----------
    if SAVE_GLOBAL_JSON:
        global_path = os.path.join(OUTPUT_DIR, "all_data.json")
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Global: {len(all_records)} samples -> {global_path}")


if __name__ == "__main__":
    main()