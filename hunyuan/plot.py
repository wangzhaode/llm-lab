import matplotlib.pyplot as plt
import numpy as np

# --- 数据定义 ---
# 定义评测基准的标签
labels = ['AIME\'24', 'AIME\'25', 'GPQA-D', 'LiveCode', 'IF-Eval', 'BFCL v3']
# 为了显示完整，将 GPQA-Diamond 缩写为 GPQA-D, LiveCodeBench 缩写为 LiveCode

# 各个级别模型的得分数据
data = {
    'Hunyuan-0.5B vs Qwen3-0.6B': {
        'Hunyuan': [17.2, 20.0, 23.3, 11.1, 49.7, 49.8],
        'Qwen3': [10.7, 15.1, 27.9, 12.3, 59.2, 46.4]
    },
    'Hunyuan-1.8B vs Qwen3-1.7B': {
        'Hunyuan': [56.7, 53.9, 47.2, 31.5, 67.6, 58.3],
        'Qwen3': [48.3, 36.8, 40.1, 33.2, 72.5, 56.6]
    },
    'Hunyuan-4B vs Qwen3-4B': {
        'Hunyuan': [78.3, 66.5, 61.1, 49.4, 76.6, 67.9],
        'Qwen3': [73.8, 65.6, 55.9, 54.5, 81.9, 65.9]
    },
    'Hunyuan-7B vs Qwen3-8B': {
        'Hunyuan': [81.1, 75.3, 60.1, 57.0, 79.3, 70.8],
        'Qwen3': [76.0, 67.3, 62.0, 57.5, 85.0, 68.1]
    }
}

# --- 绘图设置 ---

# 设置中文字体，以防图例或标题出现中文乱码
# 如果您的系统中没有 'SimHei' 字体，可以换成 'Microsoft YaHei' 或其他支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 创建一个2x2的图表网格，并设置整个图的大小
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten() # 将2x2的坐标轴数组展平，方便遍历

# 设置柱状图的宽度
bar_width = 0.35
# 为每个评测基准生成x轴的位置
x = np.arange(len(labels))

# --- 循环绘图 ---

# 遍历每个级别的数据并绘制子图
for i, (title, scores) in enumerate(data.items()):
    ax = axes[i] # 选择当前子图

    # 绘制Hunyuan的柱状图
    rects1 = ax.bar(x - bar_width/2, scores['Hunyuan'], bar_width, label='Hunyuan')
    # 绘制Qwen3的柱状图
    rects2 = ax.bar(x + bar_width/2, scores['Qwen3'], bar_width, label='Qwen3')

    # --- 美化子图 ---
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 100) # 设置Y轴范围为0-100
    ax.grid(axis='y', linestyle='--', alpha=0.7) # 添加水平网格线

    # 在每个柱子上方显示数值标签
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

# --- 全局设置 ---

# 在图表顶部添加一个统一的图例
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=14)

# 添加整个图表的大标题
# fig.suptitle('Hunyuan vs Qwen3', fontsize=20, y=1.02)

# 自动调整子图布局，防止标签重叠
plt.tight_layout(rect=[0, 0, 1, 0.95]) # rect参数为图例和标题留出空间

# --- 显示或保存图表 ---
plt.savefig('hunyuan_vs_qwen3_comparison.png', dpi=300) # 如果需要保存为图片文件，取消本行注释
plt.show() # 显示图表