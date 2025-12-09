import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ================= 配置区域 =================
MODEL_PATH = "Qwen//Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 使用与探测相同的Prompt
PROMPT = "The capital of France is"
# ===========================================

def find_top_k_super_weights(model, tokenizer, top_k=5):
    print(f"[*] Analyzing Super Weights (Top {top_k})...")

    candidates = []

    def get_analysis_hook(layer_idx):
        def hook(model, input, output):
            x_in = input[0].detach()
            w = model.weight.detach()

            # 1. 找到输出中最大的那个值 (The Super Activation)
            out_abs = output.abs()
            max_out_val = out_abs.max().item()

            # 找到最大值位置
            flat_idx = out_abs.flatten().argmax()
            seq_idx = (flat_idx // out_abs.shape[-1]) % out_abs.shape[-2]
            row_idx = (flat_idx % out_abs.shape[-1]).item()

            # 2. 回溯 Contribution
            x_vec = x_in[0, seq_idx, :]
            w_vec = w[row_idx, :]

            contributions = x_vec * w_vec
            col_idx = contributions.abs().argmax().item()
            contribution_val = contributions[col_idx].item()
            weight_val = w[row_idx, col_idx].item()

            candidates.append({
                "score": abs(contribution_val),
                "layer": layer_idx,
                "row": row_idx,
                "col": col_idx,
                "val": weight_val,
                "max_activation": max_out_val
            })

        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(get_analysis_hook(i)))

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    return sorted_candidates[:top_k]

def test_generation(model, tokenizer, prompt, description):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"\n--- {description} ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

def analyze_weight_distribution(model, layer_idx, row_idx, col_idx, sw_val):
    """
    分析 Super Weight 在其所在行中的统计地位
    """
    # 获取该层 down_proj 的完整权重矩阵 [Hidden, Intermediate]
    weight_matrix = model.model.layers[layer_idx].mlp.down_proj.weight

    # 获取特定行 (Output Channel) 的所有权重
    # 注意：down_proj 的计算是 x @ W.T，所以 W 的每一行对应一个输出维度的参数
    row_data = weight_matrix[row_idx, :].detach().float() # 转float32保证统计精度


    row_max = row_data.max().item()
    row_min = row_data.min().item()
    row_mean = row_data.mean().item()
    row_std = row_data.std().item()

    # 计算 Z-Score (偏离均值多少个标准差)
    z_score = (sw_val - row_mean) / (row_std + 1e-9)

    # 计算绝对值排名
    # argsort 默认升序，我们取反做降序
    sorted_indices = torch.argsort(row_data.abs(), descending=True)
    # 找到当前 col_idx 在排序后的位置 (Rank 1 means biggest)
    rank = (sorted_indices == col_idx).nonzero(as_tuple=True)[0].item() + 1
    total_params = len(row_data)

    print(f"  > SW Value      : {sw_val:.6f}")
    print(f"  > Row Stats     : Max={row_max:.4f}, Min={row_min:.4f}, Mean={row_mean:.6f}, Std={row_std:.4f}")
    print(f"  > Outlier Stats : Z-Score = {z_score:.2f} σ  (>3σ is rare, >10σ is extreme)")
    print(f"  > Rank In Row   : #{rank} / {total_params} (by Absolute Magnitude)")

    if rank == 1:
        print(f"  > STATUS        : It is the LARGEST weight in this row!")
    elif rank <= 10:
        print(f"  > STATUS        : It is in the Top 10 weights.")
    else:
        print(f"  > STATUS        : It is NOT a weight outlier (Magnitude is normal).")

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

    # 1. 基准测试
    test_prompt = "北京是中国的首都，"
    test_generation(model, tokenizer, test_prompt, "Original Generation")

    # 2. 寻找 Top Super Weights
    top_sws = find_top_k_super_weights(model, tokenizer, top_k=5)

    print(f"\n{'='*20} DETAILED STATISTICS {'='*20}")

    for i, sw in enumerate(top_sws):
        print(f"\n[Top-{i+1} Super Weight] Layer {sw['layer']}, Coords [{sw['row']}, {sw['col']}]")
        print(f"  > Contribution  : {sw['score']:.1f} (Activation {sw['max_activation']:.1f} x Weight {sw['val']:.4f})")

        # 执行行分布分析
        analyze_weight_distribution(model, sw['layer'], sw['row'], sw['col'], sw['val'])

    print(f"\n{'='*60}\n")

    # 3. 逐步剪枝并测试
    pruned_indices = []
    for i, sw in enumerate(top_sws):
        layer_idx = sw['layer']
        r, c = sw['row'], sw['col']

        with torch.no_grad():
            model.model.layers[layer_idx].mlp.down_proj.weight[r, c] = 0.0

        pruned_indices.append(f"L{layer_idx}")
        print(f"[*] Pruned Top-{i+1} SW at Layer {layer_idx} coords [{r},{c}]")
        test_generation(model, tokenizer, test_prompt, f"After Pruning Top {i+1} SWs")

if __name__ == "__main__":
    main()