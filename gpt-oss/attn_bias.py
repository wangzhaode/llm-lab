import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. gpt-oss-20b 的原始数据 ---
gpt_oss_20b_data = """
gpt-oss-20b
self.q_proj.bias.max = 3.875, self.k_proj.bias.max = 0.0, attn_max = 900.0
self.q_proj.bias.max = 3.015625, self.k_proj.bias.max = 0.0, attn_max = 148.0
self.q_proj.bias.max = 2.78125, self.k_proj.bias.max = 0.0, attn_max = 956.0
self.q_proj.bias.max = 2.203125, self.k_proj.bias.max = 0.0, attn_max = 160.0
self.q_proj.bias.max = 2.859375, self.k_proj.bias.max = 0.0, attn_max = 632.0
self.q_proj.bias.max = 3.5, self.k_proj.bias.max = 0.0, attn_max = 276.0
self.q_proj.bias.max = 3.015625, self.k_proj.bias.max = 0.0, attn_max = 1072.0
self.q_proj.bias.max = 2.171875, self.k_proj.bias.max = 0.0, attn_max = 318.0
self.q_proj.bias.max = 1.921875, self.k_proj.bias.max = 0.0, attn_max = 756.0
self.q_proj.bias.max = 1.7109375, self.k_proj.bias.max = 0.0, attn_max = 148.0
self.q_proj.bias.max = 2.59375, self.k_proj.bias.max = 0.0, attn_max = 243.0
self.q_proj.bias.max = 2.296875, self.k_proj.bias.max = 0.0, attn_max = 193.0
self.q_proj.bias.max = 2.328125, self.k_proj.bias.max = 0.0, attn_max = 920.0
self.q_proj.bias.max = 1.7265625, self.k_proj.bias.max = 0.0, attn_max = 224.0
self.q_proj.bias.max = 1.5546875, self.k_proj.bias.max = 0.0, attn_max = 1080.0
self.q_proj.bias.max = 1.7890625, self.k_proj.bias.max = 0.0, attn_max = 137.0
self.q_proj.bias.max = 2.078125, self.k_proj.bias.max = 0.0, attn_max = 764.0
self.q_proj.bias.max = 1.015625, self.k_proj.bias.max = 0.0, attn_max = 156.0
self.q_proj.bias.max = 1.8515625, self.k_proj.bias.max = 0.0, attn_max = 804.0
self.q_proj.bias.max = 1.046875, self.k_proj.bias.max = 0.0, attn_max = 184.0
self.q_proj.bias.max = 1.84375, self.k_proj.bias.max = 0.0, attn_max = 1512.0
self.q_proj.bias.max = 1.375, self.k_proj.bias.max = 0.0, attn_max = 179.0
self.q_proj.bias.max = 1.75, self.k_proj.bias.max = 0.0, attn_max = 1416.0
self.q_proj.bias.max = 1.6953125, self.k_proj.bias.max = 0.0, attn_max = 320.0
"""

# --- 2. 数据处理 ---
def parse_gpt_data(data_str):
    lines = data_str.strip().split('\n')[1:]
    records = []
    for i, line in enumerate(lines):
        q_k_t_max = float(line.split('attn_max = ')[1].strip())
        layer_type = "sliding_attention" if (i % 2 == 0) else "full_attention"
        records.append({
            'Layer': i + 1,
            'Layer_Type': layer_type,
            'Q_K_T_Max': q_k_t_max
        })
    return pd.DataFrame(records)

df_gpt = parse_gpt_data(gpt_oss_20b_data)

# --- 3. 最终版绘图 (修正位置) ---
sns.set_theme(style="whitegrid", context='talk')
fig, ax = plt.subplots(figsize=(18, 10))

# --- 设置标题 ---
ax.set_title("Sliding Attention operates at a fundamentally higher energy level than Full Attention",
             fontsize=16, style='italic', color='dimgray')

# --- 核心美化: 为 full_attention (谷底) 添加背景色带 ---
for i, row in df_gpt.iterrows():
    if row['Layer_Type'] == 'full_attention':
        ax.axvspan(row['Layer'] - 0.5, row['Layer'] + 0.5, color='#f0f0f0', zorder=0)

# --- 绘制数据 ---
# 1. 绘制连接线
sns.lineplot(ax=ax, data=df_gpt, x='Layer', y='Q_K_T_Max', color='black', alpha=0.4, linestyle='--', zorder=1)

# 2. 绘制数据点
palette = {
    'sliding_attention': '#c44e52',
    'full_attention': '#4c72b0'
}
markers = {
    'sliding_attention': 'o',
    'full_attention': 'X'
}

sns.scatterplot(
    ax=ax,
    data=df_gpt,
    x='Layer',
    y='Q_K_T_Max',
    hue='Layer_Type',
    style='Layer_Type',
    palette=palette,
    markers=markers,
    s=300,
    edgecolor='black',
    linewidth=1.5,
    zorder=2
)

# --- 调整坐标轴和网格 ---
ax.set_yscale('log')
ax.set_xlabel('Attention Layer', fontsize=18, weight='medium')
ax.set_ylabel('Max Q@K^T Logits (Log Scale)', fontsize=18, weight='medium')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xticks(np.arange(1, len(df_gpt) + 1, 1))
ax.grid(True, which="both", ls="--", axis='y')
ax.grid(False, axis='x')

# --- 调整图例 (移动到左下角) ---
legend = ax.legend(title="Attention Type", fontsize=16, title_fontsize=18, loc='upper left')
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_facecolor('white')

# --- 保存图像 ---
fig.tight_layout(rect=[0, 0, 1, 0.94])
file_path = 'gpt_oss_20b_final_plot_revised.png'
plt.savefig(file_path, dpi=300, bbox_inches='tight')

print(f"Final, revised analysis plot has been successfully saved to: {file_path}")