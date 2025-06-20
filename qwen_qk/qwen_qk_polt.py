import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def parse_model_data(data_str, model_name):
    lines = data_str.strip().split('\n')
    records = []
    layer_counter = 1

    for line in lines:
        line = line.strip()
        # 跳过标题行或空行
        if not line or model_name in line or "Chat" in line or "Instruct" in line:
            continue

        record = {'Model': model_name, 'Layer': layer_counter}

        # 判断是否是带有Bias的旧版模型数据
        if "q_proj.bias.max" in line:
            try:
                # 通过分割提取数据，更稳定
                parts = line.split(',')
                q_bias_str = parts[0].split('=')[1].strip()
                k_bias_str = parts[1].split('=')[1].strip()
                attn_max_str = parts[2].split('=')[1].strip()

                record['Q_Bias_Max'] = float(q_bias_str)
                record['K_Bias_Max'] = float(k_bias_str)
                record['Attn_Max'] = float(attn_max_str)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse line for {model_name}: '{line}'. Error: {e}")
                continue
        # 否则，认为是只有attn_max的新版模型数据
        else:
            try:
                attn_max_str = line.split('=')[1].strip()
                record['Q_Bias_Max'] = None
                record['K_Bias_Max'] = None
                record['Attn_Max'] = float(attn_max_str)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse line for {model_name}: '{line}'. Error: {e}")
                continue

        records.append(record)
        layer_counter += 1

    return pd.DataFrame(records)

# --- 2. 原始数据 ---
qwen_7b_data = """
Qwen-7B-Chat
self.q_proj.bias.max = 4.5, self.k_proj.bias.max = 4.53125, attn_max = 17.625
self.q_proj.bias.max = 2.15625, self.k_proj.bias.max = 4.9375, attn_max = 16.75
self.q_proj.bias.max = 2.59375, self.k_proj.bias.max = 3.90625, attn_max = 11.75
self.q_proj.bias.max = 2.765625, self.k_proj.bias.max = 3.828125, attn_max = 9.0625
self.q_proj.bias.max = 2.5625, self.k_proj.bias.max = 3.453125, attn_max = 11.75
self.q_proj.bias.max = 2.484375, self.k_proj.bias.max = 5.1875, attn_max = 10.125
self.q_proj.bias.max = 2.703125, self.k_proj.bias.max = 3.875, attn_max = 10.25
self.q_proj.bias.max = 2.21875, self.k_proj.bias.max = 4.96875, attn_max = 11.25
self.q_proj.bias.max = 2.046875, self.k_proj.bias.max = 3.25, attn_max = 19.625
self.q_proj.bias.max = 1.5234375, self.k_proj.bias.max = 2.96875, attn_max = 8.3125
self.q_proj.bias.max = 1.4453125, self.k_proj.bias.max = 1.6953125, attn_max = 5.40625
self.q_proj.bias.max = 1.53125, self.k_proj.bias.max = 1.7734375, attn_max = 6.59375
self.q_proj.bias.max = 1.609375, self.k_proj.bias.max = 5.3125, attn_max = 8.25
self.q_proj.bias.max = 1.5546875, self.k_proj.bias.max = 1.6875, attn_max = 7.5625
self.q_proj.bias.max = 1.734375, self.k_proj.bias.max = 2.40625, attn_max = 8.9375
self.q_proj.bias.max = 1.7109375, self.k_proj.bias.max = 2.078125, attn_max = 6.21875
self.q_proj.bias.max = 1.6171875, self.k_proj.bias.max = 2.3125, attn_max = 8.625
self.q_proj.bias.max = 1.65625, self.k_proj.bias.max = 2.03125, attn_max = 8.0625
self.q_proj.bias.max = 1.5625, self.k_proj.bias.max = 2.9375, attn_max = 9.0
self.q_proj.bias.max = 1.796875, self.k_proj.bias.max = 2.328125, attn_max = 8.8125
self.q_proj.bias.max = 1.765625, self.k_proj.bias.max = 2.265625, attn_max = 6.9375
self.q_proj.bias.max = 1.59375, self.k_proj.bias.max = 2.234375, attn_max = 7.6875
self.q_proj.bias.max = 1.984375, self.k_proj.bias.max = 2.671875, attn_max = 9.3125
self.q_proj.bias.max = 1.8203125, self.k_proj.bias.max = 2.046875, attn_max = 11.25
self.q_proj.bias.max = 1.640625, self.k_proj.bias.max = 2.5, attn_max = 7.03125
self.q_proj.bias.max = 1.9140625, self.k_proj.bias.max = 2.65625, attn_max = 6.96875
self.q_proj.bias.max = 1.9140625, self.k_proj.bias.max = 2.953125, attn_max = 9.4375
self.q_proj.bias.max = 1.921875, self.k_proj.bias.max = 2.234375, attn_max = 7.8125
self.q_proj.bias.max = 1.8671875, self.k_proj.bias.max = 2.34375, attn_max = 9.875
self.q_proj.bias.max = 1.84375, self.k_proj.bias.max = 2.125, attn_max = 10.875
self.q_proj.bias.max = 1.8203125, self.k_proj.bias.max = 3.984375, attn_max = 12.4375
self.q_proj.bias.max = 1.2890625, self.k_proj.bias.max = 10.75, attn_max = 35.25
"""

qwen1_5_7b_data = """
Qwen1.5-7B-Chat
self.q_proj.bias.max = 4.3125, self.k_proj.bias.max = 4.71875, attn_max = 22.5
self.q_proj.bias.max = 2.203125, self.k_proj.bias.max = 5.3125, attn_max = 18.25
self.q_proj.bias.max = 2.65625, self.k_proj.bias.max = 4.21875, attn_max = 13.625
self.q_proj.bias.max = 2.8125, self.k_proj.bias.max = 4.03125, attn_max = 9.125
self.q_proj.bias.max = 2.625, self.k_proj.bias.max = 3.484375, attn_max = 15.0
self.q_proj.bias.max = 2.546875, self.k_proj.bias.max = 6.5, attn_max = 9.25
self.q_proj.bias.max = 2.75, self.k_proj.bias.max = 4.5, attn_max = 9.9375
self.q_proj.bias.max = 2.265625, self.k_proj.bias.max = 5.84375, attn_max = 8.875
self.q_proj.bias.max = 2.078125, self.k_proj.bias.max = 3.359375, attn_max = 10.9375
self.q_proj.bias.max = 1.546875, self.k_proj.bias.max = 4.25, attn_max = 6.84375
self.q_proj.bias.max = 1.4453125, self.k_proj.bias.max = 2.015625, attn_max = 7.21875
self.q_proj.bias.max = 1.515625, self.k_proj.bias.max = 2.5625, attn_max = 6.59375
self.q_proj.bias.max = 1.5859375, self.k_proj.bias.max = 5.5625, attn_max = 9.4375
self.q_proj.bias.max = 1.515625, self.k_proj.bias.max = 2.3125, attn_max = 7.8125
self.q_proj.bias.max = 1.71875, self.k_proj.bias.max = 2.5625, attn_max = 12.625
self.q_proj.bias.max = 1.671875, self.k_proj.bias.max = 2.484375, attn_max = 7.875
self.q_proj.bias.max = 1.5546875, self.k_proj.bias.max = 3.828125, attn_max = 9.4375
self.q_proj.bias.max = 1.6484375, self.k_proj.bias.max = 2.09375, attn_max = 8.625
self.q_proj.bias.max = 1.53125, self.k_proj.bias.max = 3.90625, attn_max = 13.4375
self.q_proj.bias.max = 1.7265625, self.k_proj.bias.max = 3.53125, attn_max = 8.8125
self.q_proj.bias.max = 1.734375, self.k_proj.bias.max = 2.53125, attn_max = 10.3125
self.q_proj.bias.max = 1.5625, self.k_proj.bias.max = 2.421875, attn_max = 10.25
self.q_proj.bias.max = 1.921875, self.k_proj.bias.max = 2.546875, attn_max = 10.8125
self.q_proj.bias.max = 1.78125, self.k_proj.bias.max = 2.15625, attn_max = 10.3125
self.q_proj.bias.max = 1.6171875, self.k_proj.bias.max = 2.421875, attn_max = 8.8125
self.q_proj.bias.max = 1.8828125, self.k_proj.bias.max = 2.5625, attn_max = 10.8125
self.q_proj.bias.max = 1.9453125, self.k_proj.bias.max = 2.875, attn_max = 11.9375
self.q_proj.bias.max = 1.8984375, self.k_proj.bias.max = 2.40625, attn_max = 12.375
self.q_proj.bias.max = 1.828125, self.k_proj.bias.max = 2.4375, attn_max = 11.9375
self.q_proj.bias.max = 1.796875, self.k_proj.bias.max = 2.453125, attn_max = 15.375
self.q_proj.bias.max = 1.78125, self.k_proj.bias.max = 4.5, attn_max = 13.4375
self.q_proj.bias.max = 1.2578125, self.k_proj.bias.max = 11.625, attn_max = 33.5
"""

qwen2_7b_data = """
Qwen2-7B-Instruct
self.q_proj.bias.max = 44.75, self.k_proj.bias.max = 166.0, attn_max = 3728.0
self.q_proj.bias.max = 6.125, self.k_proj.bias.max = 39.5, attn_max = 59.5
self.q_proj.bias.max = 15.125, self.k_proj.bias.max = 28.5, attn_max = 26.375
self.q_proj.bias.max = 13.4375, self.k_proj.bias.max = 68.0, attn_max = 31.25
self.q_proj.bias.max = 23.375, self.k_proj.bias.max = 2.421875, attn_max = 20.25
self.q_proj.bias.max = 14.875, self.k_proj.bias.max = 6.375, attn_max = 8.4375
self.q_proj.bias.max = 33.0, self.k_proj.bias.max = 16.25, attn_max = 23.5
self.q_proj.bias.max = 16.125, self.k_proj.bias.max = 5.875, attn_max = 13.375
self.q_proj.bias.max = 20.75, self.k_proj.bias.max = 3.953125, attn_max = 5.625
self.q_proj.bias.max = 13.0, self.k_proj.bias.max = 5.59375, attn_max = 8.0625
self.q_proj.bias.max = 25.25, self.k_proj.bias.max = 4.53125, attn_max = 16.125
self.q_proj.bias.max = 15.1875, self.k_proj.bias.max = 5.0625, attn_max = 3.46875
self.q_proj.bias.max = 22.625, self.k_proj.bias.max = 16.625, attn_max = 28.5
self.q_proj.bias.max = 21.375, self.k_proj.bias.max = 39.75, attn_max = 28.125
self.q_proj.bias.max = 18.75, self.k_proj.bias.max = 15.0, attn_max = 36.0
self.q_proj.bias.max = 14.375, self.k_proj.bias.max = 23.75, attn_max = 29.0
self.q_proj.bias.max = 12.0625, self.k_proj.bias.max = 15.75, attn_max = 27.625
self.q_proj.bias.max = 13.4375, self.k_proj.bias.max = 7.59375, attn_max = 25.5
self.q_proj.bias.max = 31.25, self.k_proj.bias.max = 4.625, attn_max = 75.5
self.q_proj.bias.max = 44.75, self.k_proj.bias.max = 12.0625, attn_max = 22.375
self.q_proj.bias.max = 17.5, self.k_proj.bias.max = 4.3125, attn_max = 13.625
self.q_proj.bias.max = 13.75, self.k_proj.bias.max = 5.125, attn_max = 5.40625
self.q_proj.bias.max = 12.75, self.k_proj.bias.max = 5.65625, attn_max = 13.375
self.q_proj.bias.max = 11.5, self.k_proj.bias.max = 4.0, attn_max = 8.4375
self.q_proj.bias.max = 24.0, self.k_proj.bias.max = 4.625, attn_max = 13.9375
self.q_proj.bias.max = 9.6875, self.k_proj.bias.max = 8.9375, attn_max = 11.375
self.q_proj.bias.max = 15.5625, self.k_proj.bias.max = 3.53125, attn_max = 6.59375
self.q_proj.bias.max = 46.0, self.k_proj.bias.max = 226.0, attn_max = 1888.0
"""

qwen2_5_7b_data = """
Qwen2.5-7B-Instruct
self.q_proj.bias.max = 46.25, self.k_proj.bias.max = 171.0, attn_max = 3600.0
self.q_proj.bias.max = 6.03125, self.k_proj.bias.max = 62.5, attn_max = 452.0
self.q_proj.bias.max = 15.125, self.k_proj.bias.max = 27.875, attn_max = 25.25
self.q_proj.bias.max = 13.0, self.k_proj.bias.max = 68.5, attn_max = 39.0
self.q_proj.bias.max = 23.375, self.k_proj.bias.max = 2.390625, attn_max = 21.5
self.q_proj.bias.max = 15.125, self.k_proj.bias.max = 6.15625, attn_max = 9.0625
self.q_proj.bias.max = 33.0, self.k_proj.bias.max = 16.125, attn_max = 26.125
self.q_proj.bias.max = 16.25, self.k_proj.bias.max = 5.5625, attn_max = 15.4375
self.q_proj.bias.max = 20.875, self.k_proj.bias.max = 3.890625, attn_max = 6.0
self.q_proj.bias.max = 13.0, self.k_proj.bias.max = 5.40625, attn_max = 7.125
self.q_proj.bias.max = 25.25, self.k_proj.bias.max = 4.5, attn_max = 18.875
self.q_proj.bias.max = 15.125, self.k_proj.bias.max = 5.0, attn_max = 4.09375
self.q_proj.bias.max = 22.625, self.k_proj.bias.max = 16.25, attn_max = 28.5
self.q_proj.bias.max = 21.5, self.k_proj.bias.max = 39.5, attn_max = 29.875
self.q_proj.bias.max = 18.75, self.k_proj.bias.max = 15.1875, attn_max = 42.5
self.q_proj.bias.max = 14.375, self.k_proj.bias.max = 23.75, attn_max = 28.125
self.q_proj.bias.max = 12.1875, self.k_proj.bias.max = 15.5625, attn_max = 32.25
self.q_proj.bias.max = 13.4375, self.k_proj.bias.max = 7.25, attn_max = 28.25
self.q_proj.bias.max = 30.625, self.k_proj.bias.max = 4.5625, attn_max = 74.0
self.q_proj.bias.max = 44.75, self.k_proj.bias.max = 11.75, attn_max = 26.375
self.q_proj.bias.max = 17.625, self.k_proj.bias.max = 4.0, attn_max = 12.5625
self.q_proj.bias.max = 13.8125, self.k_proj.bias.max = 5.0625, attn_max = 4.59375
self.q_proj.bias.max = 12.875, self.k_proj.bias.max = 5.46875, attn_max = 23.0
self.q_proj.bias.max = 11.625, self.k_proj.bias.max = 3.734375, attn_max = 10.8125
self.q_proj.bias.max = 24.0, self.k_proj.bias.max = 4.5625, attn_max = 15.125
self.q_proj.bias.max = 9.6875, self.k_proj.bias.max = 8.3125, attn_max = 14.5
self.q_proj.bias.max = 15.5625, self.k_proj.bias.max = 3.4375, attn_max = 7.8125
self.q_proj.bias.max = 46.75, self.k_proj.bias.max = 233.0, attn_max = 888.0
"""

qwen3_8b_data = """
Qwen3-8B (no qk_bias)
attn_max = 19.75
attn_max = 31.125
attn_max = 24.25
attn_max = 32.75
attn_max = 16.875
attn_max = 26.5
attn_max = 16.25
attn_max = 47.0
attn_max = 18.375
attn_max = 23.0
attn_max = 17.0
attn_max = 18.875
attn_max = 19.125
attn_max = 25.75
attn_max = 18.0
attn_max = 22.125
attn_max = 20.0
attn_max = 27.0
attn_max = 17.875
attn_max = 24.75
attn_max = 27.875
attn_max = 20.75
attn_max = 26.75
attn_max = 25.625
attn_max = 25.25
attn_max = 22.5
attn_max = 22.375
attn_max = 19.5
attn_max = 18.875
attn_max = 21.0
attn_max = 20.875
attn_max = 20.125
attn_max = 20.0
attn_max = 19.25
attn_max = 19.375
attn_max = 15.1875
"""

# --- 3. 数据处理 ---
df_qwen = parse_model_data(qwen_7b_data, 'Qwen-7B')
df_qwen1_5 = parse_model_data(qwen1_5_7b_data, 'Qwen1.5-7B')
df_qwen2 = parse_model_data(qwen2_7b_data, 'Qwen2-7B')
df_qwen2_5 = parse_model_data(qwen2_5_7b_data, 'Qwen2.5-7B')
df_qwen3 = parse_model_data(qwen3_8b_data, 'Qwen3-8B')

# 合并所有数据到一个DataFrame
df_all = pd.concat([df_qwen, df_qwen1_5, df_qwen2, df_qwen2_5, df_qwen3], ignore_index=True)

# 方便绘图，计算Bias的最大值
df_all['Bias_Max'] = df_all[['Q_Bias_Max', 'K_Bias_Max']].max(axis=1)

# --- 4. 绘图与分析 ---
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# 设置绘图风格
sns.set_theme(style="whitegrid", palette="viridis")

# 创建一个Figure对象
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Analysis of Qwen Attention Mechanism Evolution: From Bias to Norm', fontsize=22, weight='bold')

# 使用GridSpec来创建更灵活的布局
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# 定义模型顺序
model_order = ['Qwen-7B', 'Qwen1.5-7B', 'Qwen2-7B', 'Qwen2.5-7B', 'Qwen3-8B']

# --- 图1 (左上角): Attention Score 最大值的演进 ---
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(ax=ax1, x='Model', y='Attn_Max', data=df_all, order=model_order)
ax1.set_yscale('log')

ax1.set_title('A: Evolution of Max Attention Score (Q@K^T)', fontsize=16, weight='bold', loc='left')
ax1.set_xlabel('')
ax1.set_ylabel('Max Q@K^T Value (Log Scale)', fontsize=12)
ax1.tick_params(axis='x', rotation=15)

# --- 图2 (右上角): QK-Bias 最大值的演进 ---
ax2 = fig.add_subplot(gs[0, 1])
df_with_bias = df_all.dropna(subset=['Bias_Max'])
bias_model_order = ['Qwen-7B', 'Qwen1.5-7B', 'Qwen2-7B', 'Qwen2.5-7B']
sns.boxplot(ax=ax2, x='Model', y='Bias_Max', data=df_with_bias, order=bias_model_order)
ax2.set_yscale('log')

ax2.set_title('B: Evolution of Max QK-Bias', fontsize=16, weight='bold', loc='left')
ax2.set_xlabel('')
ax2.set_ylabel('Max QK-Bias Value (Log Scale)', fontsize=12)
ax2.tick_params(axis='x', rotation=15)

# --- 图3 (占据下方整行): 分层展示Attn_Max的变化 ---
ax3 = fig.add_subplot(gs[1, :]) # gs[1, :] 表示占据第1行（从0开始）的所有列
palette = sns.color_palette("viridis", n_colors=len(df_all['Model'].unique()))
hue_order = model_order
sns.lineplot(ax=ax3, x='Layer', y='Attn_Max', hue='Model', data=df_all,
             palette=palette, hue_order=hue_order, legend='full', marker='o', markersize=5, alpha=0.7)
ax3.set_yscale('log')
max_layer = df_all['Layer'].max()
tick_step = 5  # 每隔5个layer显示一个刻度
x_ticks = np.arange(1, max_layer + 1, tick_step)
if max_layer not in x_ticks:
    x_ticks = np.append(x_ticks, max_layer)
ax3.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
ax3.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax3.xaxis.set_minor_locator(mticker.NullLocator())
ax3.set_xlim(0, max_layer + 1) # 设置X轴范围，留出一些边距
ax3.set_title('C: Per-Layer Max Attention Score Comparison', fontsize=16, weight='bold', loc='left')
ax3.set_xlabel('Model Layer Index', fontsize=12)
ax3.set_ylabel('Max Q@K^T Value (Log Scale)', fontsize=12)
ax3.legend(title='Model Version')
ax3.grid(True, which="both", ls="--")

# --- 保存图像 ---
file_path = 'qwen_qk_evolution_analysis.png'
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Final corrected analysis plot has been successfully saved to: {file_path}")