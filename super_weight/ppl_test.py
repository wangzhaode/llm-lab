import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm

# ================= 配置区域 =================
MODEL_PATH = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "The capital of France is"

# 量化配置
QUANT_BITS = 4
GROUP_SIZE = 64
# ===========================================

def get_wikitext2_data(tokenizer):
    """加载 WikiText-2 测试集"""
    print("[*] Loading WikiText-2 dataset...")
    try:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except:
        print("Failed to load via HF datasets, trying offline or skipping...")
        return None

    encodings = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return encodings

def evaluate_perplexity(model, encodings, stride=512):
    """计算 PPL (Perplexity)"""
    print("[*] Evaluating Perplexity...")
    model.eval()
    max_length = 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    # 为了速度，测试前 50000 个 token
    limit = min(seq_len, 30000)

    for begin_loc in tqdm(range(0, limit, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def pseudo_quantize_tensor(w, bits=4, group_size=128):
    """
    模拟 INT4 非对称量化 (Fake Quantization)
    """
    org_shape = w.shape
    device = w.device # 保持设备一致

    # Reshape into groups
    if w.numel() % group_size != 0:
        # 简单padding处理
        pad_len = group_size - (w.numel() % group_size)
        w_flat = torch.cat([w.flatten(), torch.zeros(pad_len, device=device)])
        w_flat = w_flat.reshape(-1, group_size)
        padding = True
    else:
        w_flat = w.reshape(-1, group_size)
        padding = False

    # 1. 计算 Scale 和 ZeroPoint
    max_val = w_flat.amax(dim=1, keepdim=True)
    min_val = w_flat.amin(dim=1, keepdim=True)
    max_val = torch.clamp(max_val, min=0)
    min_val = torch.clamp(min_val, max=0)

    scale = (max_val - min_val) / (2 ** bits - 1)
    scale = torch.clamp(scale, min=1e-5)

    zero_point = -min_val / scale
    zero_point = torch.round(zero_point)

    # 2. Quantize
    w_q = torch.clamp(torch.round(w_flat / scale) + zero_point, 0, 2**bits - 1)

    # 3. Dequantize
    w_dq = (w_q - zero_point) * scale

    # Restore shape
    w_dq = w_dq.flatten()
    if padding:
        w_dq = w_dq[:w.numel()]

    return w_dq.reshape(org_shape)

def find_top_k_super_weights(model, tokenizer, top_k=5):
    print(f"[*] Analyzing Super Weights (Top {top_k})...")
    candidates = []

    def get_analysis_hook(layer_idx):
        def hook(model, input, output):
            x_in = input[0].detach()
            w = model.weight.detach()
            out_abs = output.abs()

            flat_idx = out_abs.flatten().argmax()
            row_idx = (flat_idx % out_abs.shape[-1]).item()
            seq_idx = (flat_idx // out_abs.shape[-1]) % out_abs.shape[-2]

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
                "val": weight_val
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

def run_quantization_experiment(model, tokenizer, super_weights):

    wiki_data = get_wikitext2_data(tokenizer)
    if wiki_data is None: return

    # 1. Baseline
    print("\n=== Experiment 1: FP16 Baseline ===")
    ppl_fp16 = evaluate_perplexity(model, wiki_data)
    print(f"Result: FP16 PPL = {ppl_fp16:.4f}")

    print("[*] Backing up weights...")
    original_weights = {}
    for i, layer in enumerate(model.model.layers):
        # 备份到 CPU
        original_weights[i] = layer.mlp.down_proj.weight.detach().clone().cpu()

    # 2. Naive INT4
    print("\n=== Experiment 2: Naive INT4 Quantization ===")
    print(f"[*] Simulating INT{QUANT_BITS} GroupSize={GROUP_SIZE} on all down_proj layers...")

    for i, layer in enumerate(model.model.layers):
        w = layer.mlp.down_proj.weight.data
        w_quant = pseudo_quantize_tensor(w, bits=QUANT_BITS, group_size=GROUP_SIZE)
        layer.mlp.down_proj.weight.data = w_quant

    ppl_naive = evaluate_perplexity(model, wiki_data)
    print(f"Result: Naive INT4 PPL = {ppl_naive:.4f}")

    # 恢复权重 - 【修复点】
    print("[*] Restoring weights for next experiment...")
    for i, layer in enumerate(model.model.layers):
        # 获取该层当前的 device
        target_device = layer.mlp.down_proj.weight.device
        # 将备份的权重移动到该层对应的 device
        layer.mlp.down_proj.weight.data = original_weights[i].to(target_device)

    # 3. SW-Aware INT4
    print("\n=== Experiment 3: Super Weight Aware INT4 (Clip & Restore) ===")
    print("[*] Applying 'Clip & Restore' strategy...")

    sw_lookup = {}
    for sw in super_weights:
        l, r, c = sw['layer'], sw['row'], sw['col']
        # 从备份中读取原始值
        orig_val = original_weights[l][r, c].item()
        sw_lookup[(l,r,c)] = orig_val
        # Clip: 设为 0
        model.model.layers[l].mlp.down_proj.weight.data[r, c] = 0.0

    for i, layer in enumerate(model.model.layers):
        w = layer.mlp.down_proj.weight.data
        w_quant = pseudo_quantize_tensor(w, bits=QUANT_BITS, group_size=GROUP_SIZE)
        layer.mlp.down_proj.weight.data = w_quant

    # Restore Phase
    count_restored = 0
    for (l, r, c), val in sw_lookup.items():
        model.model.layers[l].mlp.down_proj.weight.data[r, c] = val
        count_restored += 1

    print(f"[*] Restored {count_restored} Super Weights to FP16.")

    ppl_aware = evaluate_perplexity(model, wiki_data)
    print(f"Result: SW-Aware INT4 PPL = {ppl_aware:.4f}")

    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: INT{QUANT_BITS}, GroupSize={GROUP_SIZE}")
    print("-" * 40)
    print(f"FP16 Baseline PPL      : {ppl_fp16:.4f}")
    print(f"Naive INT4 PPL         : {ppl_naive:.4f}")
    print(f"SW-Aware INT4 PPL      : {ppl_aware:.4f}")
    print(f"Improvement            : {ppl_naive - ppl_aware:.4f}")
    print("="*40)

def main():
    print(f"[*] Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # device_map="auto" 会导致模型分布在多卡上
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16
    )

    top_sws = find_top_k_super_weights(model, tokenizer, top_k=20)

    print("\n[Identified Super Weights Sample]")
    for i, sw in enumerate(top_sws[:3]):
        print(f"Rank {i+1}: Layer {sw['layer']}, Coords [{sw['row']},{sw['col']}], Score {sw['score']:.1f}")

    run_quantization_experiment(model, tokenizer, top_sws)

if __name__ == "__main__":
    main()