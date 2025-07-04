import json
import glob
import matplotlib.pyplot as plt

def plot_qkv_bias_max_comparison(model_data_paths: dict):
    """
    加载多个模型的 QKV bias max 数据，并绘制对比图。

    Args:
        model_data_paths (dict): 字典，键为模型名称，值为对应 JSON 文件的路径。
                                 例如：{"Pangu": "pangu_qkv_bias_max.json", "Qwen2.5-14B": "qwen2_5_14b_qkv_bias_max.json"}
    """
    plt.figure(figsize=(15, 12))

    for proj_type in ['q', 'k', 'v']:
        plt.subplot(3, 1, {'q':1, 'k':2, 'v':3}[proj_type]) # 3行1列的子图

        for model_name, json_path in model_data_paths.items():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                layers = []
                bias_max_values = []
                for layer_key in data:
                    layer_idx = int(layer_key.split('_')[1])
                    key_name = f"{proj_type}_proj_bias_max"
                    if key_name in data[layer_key]:
                        layers.append(layer_idx)
                        bias_max_values.append(data[layer_key][key_name])

                # 按照层索引排序
                sorted_pairs = sorted(zip(layers, bias_max_values))
                sorted_layers = [pair[0] for pair in sorted_pairs]
                sorted_values = [pair[1] for pair in sorted_pairs]

                plt.plot(sorted_layers, sorted_values, label=model_name)
            except FileNotFoundError:
                print(f"警告: 未找到文件 '{json_path}'，跳过。")
            except json.JSONDecodeError:
                print(f"错误: 文件 '{json_path}' 不是有效的 JSON 格式。")
            except Exception as e:
                print(f"处理 '{json_path}' 时发生未知错误: {e}")

        plt.title(f'{proj_type.upper()} Projection Bias Max')
        plt.xlabel('Layer')
        plt.ylabel('Max Bias Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    plt.tight_layout()
    plt.savefig('./compare_qkv_bias_max.png', dpi=300)

if __name__ == "__main__":
    files = glob.glob('./bias/*.json')
    model_data_for_plot = {}
    for file in files:
        model_name = file.split('/')[-1].split('_')[0]
        model_data_for_plot[model_name] = file

    plot_qkv_bias_max_comparison(model_data_for_plot)