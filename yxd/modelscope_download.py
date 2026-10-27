from modelscope import dataset_snapshot_download

# 自定义存放目录
TARGET_LOCAL_DIR = "/root/yxd/Diff4Qwen/data/parquet_data"

dataset_path = dataset_snapshot_download(
    dataset_id="artplus/PrismLayersPro",
    local_dir=TARGET_LOCAL_DIR,
    allow_patterns=["data/*.parquet"]  # 精准匹配data下所有furry开头parquet
)
print("数据集下载完成，路径：", dataset_path)