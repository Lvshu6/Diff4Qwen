import os
import json
import glob

# ================= 配置区 =================
# 设置你的目标文件夹路径（根据你的图片，路径如下）
# 如果脚本和exported在同一级目录，可以这样写：
TARGET_DIR = "/root/yxd/Diff4Qwen/data/parquet_data/export"

# 设置输出文件的路径和名称
OUTPUT_FILE = os.path.join(TARGET_DIR, "merged_output.json")
# =========================================

def merge_json_files(target_dir, output_file):
    # 1. 查找所有子文件夹
    # os.path.join 会拼凑出 000000, 000001 等文件夹路径
    sub_dirs = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    
    print(f"找到 {len(sub_dirs)} 个子文件夹。")
    
    merged_list = []
    
    # 2. 遍历每个子文件夹
    for dir_name in sub_dirs:
        dir_path = os.path.join(target_dir, dir_name)
        
        # 3. 查找该文件夹下的 json 文件
        # 使用 glob 匹配 .json 文件
        json_files = glob.glob(os.path.join(dir_path, "*.json"))
        
        # 通常一个文件夹只有一个 json，我们默认处理第一个
        if len(json_files) == 0:
            print(f"警告: 文件夹 {dir_name} 中没有找到 json 文件，跳过。")
            continue
            
        json_path = json_files[0]
        
        try:
            # 4. 读取并追加数据
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 假设 json 文件里存的是一个列表（如你提供的示例）
            if isinstance(data, list):
                merged_list.extend(data)
            else:
                # 如果 json 里是一个字典对象，将其包裹在列表中再添加
                merged_list.append(data)
                
            print(f"已处理: {json_path}")
            
        except json.JSONDecodeError:
            print(f"错误: {json_path} 不是有效的 JSON 格式，跳过。")
        except Exception as e:
            print(f"处理 {json_path} 时出错: {e}")

    # 5. 保存合并后的结果
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_list, f_out, indent=2, ensure_ascii=False)
        
    print("-" * 30)
    print(f"合并完成！")
    print(f"总共合并了 {len(merged_list)} 条数据。")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    merge_json_files(TARGET_DIR, OUTPUT_FILE)