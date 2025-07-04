import os
import json
import re
import glob
import numpy as np
from safetensors import safe_open
import logging

# 配置日志，方便调试和查看进度
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_qkv_bias_max(model_dir: str, output_filename: str = "qkv_bias_max.json"):
    """
    遍历指定模型目录下的 safetensors 文件，提取每一层 Q, K, V 投影的偏置的最大值。
    将结果保存到 JSON 文件中。

    Args:
        model_dir (str): 包含 safetensors 文件的模型目录路径。
        output_filename (str): 输出 JSON 文件的名称。该文件将保存在 model_dir 中。
    """
    if not os.path.isdir(model_dir):
        logging.error(f"错误：模型目录 '{model_dir}' 不存在或不是一个目录。")
        return

    # 存储结果的字典，格式为 { "layer_0": { "q_proj_bias_max": val, ... }, ... }
    results = {}

    # 正则表达式用于匹配 QKV 偏置的名称，并提取层索引和投影类型 (q, k, v)
    # 常见的命名模式：
    # - model.layers.0.self_attn.q_proj.bias
    # - transformer.h.1.attn.k_proj.bias
    # - base_model.model.layers.2.self_attn.v_proj.bias
    # 确保能够捕获层索引（\d+）和投影类型（q|k|v）
    qkv_bias_pattern = re.compile(r'.*(?:layers|h)\.(\d+)\..*.(q|k|v)_proj\.bias$')

    # 查找目录中的所有 .safetensors 文件
    safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]

    if not safetensors_files:
        logging.warning(f"在目录 '{model_dir}' 中未找到任何 .safetensors 文件。")
        return

    logging.info(f"在 '{model_dir}' 中找到 {len(safetensors_files)} 个 safetensors 文件，开始处理...")

    for filename in safetensors_files:
        file_path = os.path.join(model_dir, filename)
        logging.info(f"正在处理文件：'{filename}'...")
        try:
            # 使用 safetensors.safe_open 打开文件，并指定 framework="pt" 以确保加载为 PyTorch 张量
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    match = qkv_bias_pattern.match(key)
                    if match:
                        layer_idx_str, proj_type = match.groups()
                        layer_idx = int(layer_idx_str)
                        proj_name_key = f"{proj_type}_proj_bias_max"
                        layer_key = f"layer_{layer_idx}"

                        try:
                            # 获取张量并计算其最大值
                            tensor = f.get_tensor(key)
                            max_val = tensor.max().item()

                            # 将结果存储到 results 字典中
                            if layer_key not in results:
                                results[layer_key] = {}
                            results[layer_key][proj_name_key] = max_val
                            # logging.debug(f"  发现 {key}: 偏置最大值 = {max_val}")
                        except Exception as e:
                            logging.error(f"  处理张量 '{key}' 时发生错误：{e}")
        except Exception as e:
            logging.error(f"  打开或读取文件 '{filename}' 时发生错误：{e}")

    # 根据层索引对结果进行排序，以便输出更规整
    # 'layer_0', 'layer_10', 'layer_1' 这样的字符串排序是不对的，需要按数字排序
    sorted_results = dict(sorted(results.items(), key=lambda item: int(item[0].split('_')[1])))

    output_path = './bias/' + output_filename
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)
        logging.info(f"结果已成功保存到 '{output_path}'")
    except Exception as e:
        logging.error(f"保存结果到 '{output_path}' 时发生错误：{e}")

# --- 示例用法 ---
if __name__ == "__main__":
    # models = glob.glob('/home/yanxing/data/models/*Qwen2.5-*7B*')
    models = [
        '/home/yanxing/data/models/Qwen2.5-14B-Instruct',
        '/home/yanxing/data/models/Qwen2.5-14B-DeepSeek-R1-1M',
        '/home/yanxing/data/models/pangu-pro-moe-model'
    ]
    for model_dir in models:
        res_json = os.path.basename(model_dir) + "_qkv_bias_max.json"
        extract_qkv_bias_max(model_dir, res_json)
