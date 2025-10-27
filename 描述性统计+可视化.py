import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
import os

warnings.filterwarnings("ignore")

# ---------------------- 基础设置（重点解决中文显示） ----------------------
# 手动指定系统中存在的中文字体（确保路径正确，以下为Windows常见字体路径示例）
try:
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
except:
    # 若指定字体失败， fallback到系统可用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use('seaborn-v0_8-ticks')
VISUAL_DIR = "描述性统计/"


def descriptive_analysis(static_path, dynamic_path):
    if not os.path.exists(VISUAL_DIR):
        os.makedirs(VISUAL_DIR)

    static_df = pd.read_csv(static_path, encoding="utf-8-sig")
    dynamic_df = pd.read_csv(dynamic_path, encoding="utf-8-sig")

    numeric_cols = ["视频时长（秒）", "当前点赞量", "当前播放量", "当前分享量"]
    for col in numeric_cols:
        if col in static_df.columns:
            static_df[col] = pd.to_numeric(static_df[col], errors="coerce")
        if col in dynamic_df.columns:
            dynamic_df[col] = pd.to_numeric(dynamic_df[col], errors="coerce")

    # ---------------------- 核心指标统计特征 ----------------------
    print("=" * 50)
    print("2. 核心指标统计特征")
    print("=" * 50)
    duration_stats = static_df["视频时长（秒）"].describe().round(2)
    print("【视频时长（秒）】")
    print(
        f"均值：{duration_stats['mean']} | 中位数：{duration_stats['50%']} | 最小值：{duration_stats['min']} | 最大值：{duration_stats['max']}")
    print(f"标准差：{duration_stats['std']} | 25分位数：{duration_stats['25%']} | 75分位数：{duration_stats['75%']}\n")

    dynamic_df["历史状态长度"] = dynamic_df["历史状态"].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
    last_state_dynamic = dynamic_df.loc[dynamic_df.groupby("url")["历史状态长度"].idxmax()]

    interaction_cols = ["当前点赞量", "当前播放量", "当前分享量"]
    interaction_stats = last_state_dynamic[interaction_cols].describe().round(2)
    for col in interaction_cols:
        print(f"【{col}】")
        print(f"均值：{interaction_stats.loc['mean', col]} | 中位数：{interaction_stats.loc['50%', col]}")
        print(f"最小值：{interaction_stats.loc['min', col]} | 最大值：{interaction_stats.loc['max', col]}")
        print(
            f"标准差：{interaction_stats.loc['std', col]} | 25分位数：{interaction_stats.loc['25%', col]} | 75分位数：{interaction_stats.loc['75%', col]}\n")

    excel_path = f"{VISUAL_DIR}核心指标统计结果.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(duration_stats).T.to_excel(writer, sheet_name="视频时长统计")
        interaction_stats.to_excel(writer, sheet_name="互动指标统计")
    print(f"→ 统计结果Excel已保存至：{excel_path}\n")

    # ---------------------- 可视化图表 ----------------------
    print("=" * 50)
    print("3. 生成可视化图表")
    print("=" * 50)

    # 3.1 箱线图：核心互动指标分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    interaction_labels = ["点赞量", "播放量", "分享量"]
    for i, (col, label) in enumerate(zip(interaction_cols, interaction_labels)):
        q95 = last_state_dynamic[col].quantile(0.95)
        data = last_state_dynamic[last_state_dynamic[col] <= q95][col]

        bp = axes[i].boxplot(data, patch_artist=True, medianprops={"color": "#E74C3C", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#3498DB")
        bp["boxes"][0].set_alpha(0.7)

        axes[i].set_title(f"{label}分布（95分位数以内）", fontsize=12, fontweight="bold", fontproperties=font_prop)
        axes[i].set_ylabel(label, fontsize=10, fontproperties=font_prop)
        axes[i].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{VISUAL_DIR}互动指标箱线图.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("→ 互动指标箱线图已保存")



    # 3.2 热力图：核心变量间相关性
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_vars = ["视频时长（秒）", "当前点赞量", "当前播放量", "当前分享量"]
    corr_data = pd.merge(
        static_df[["url", "视频时长（秒）"]],
        last_state_dynamic[["url", "当前点赞量", "当前播放量", "当前分享量"]],
        on="url",
        how="inner"
    ).dropna(subset=corr_vars)

    corr_matrix = corr_data[corr_vars].corr().round(3)
    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)

    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            text = ax.text(j, i, corr_matrix.iloc[i, j],
                           ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(corr_vars)))
    ax.set_yticks(range(len(corr_vars)))
    ax.set_xticklabels(["视频时长", "点赞量", "播放量", "分享量"], rotation=45, fontproperties=font_prop)
    ax.set_yticklabels(["视频时长", "点赞量", "播放量", "分享量"], fontproperties=font_prop)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson相关系数", rotation=270, labelpad=20, fontsize=12, fontproperties=font_prop)

    ax.set_title("核心变量相关性热力图", fontsize=14, fontweight="bold", pad=20, fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(f"{VISUAL_DIR}变量相关性热力图.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("→ 变量相关性热力图已保存")

    print(f"\n✅ 所有图表和统计结果已保存至“描述性统计”文件夹！")


if __name__ == "__main__":
    CLEANED_STATIC_PATH = "video_static_2.csv"
    CLEANED_DYNAMIC_PATH = "video_dynamic_2.csv"
    descriptive_analysis(CLEANED_STATIC_PATH, CLEANED_DYNAMIC_PATH)