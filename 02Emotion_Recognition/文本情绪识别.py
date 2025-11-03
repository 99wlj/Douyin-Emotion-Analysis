import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tqdm import tqdm

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def predict_emotion(merged_text_path, model_name = "hfl/chinese-bert-wwm"):
    """
    用预训练模型预测文本情绪，输出sentiment_score和emotion_label
    :param merged_text_path: 合并文本表路径（含url、text等）
    :param model_name: 预训练模型名称
    :return: 带情绪标签的DataFrame
    """
    # 1. 读取合并文本数据
    merged_df = pd.read_csv(merged_text_path, encoding="utf-8-sig")
    print(f"加载合并文本数据完成，共{len(merged_df)}条记录")

    # 2. 加载预训练模型和分词器（中文BERT-wwm）
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类：积极/消极

    # 3. 构建情绪分析pipeline（支持批量处理）
    device = 0 if torch.cuda.is_available() else -1  # 自动判断是否使用GPU
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,  # 自动截断过长文本
        max_length=512  # BERT最大输入长度
    )

    # 4. 批量预测情绪（分块处理，避免内存溢出）
    batch_size = 32  # 每批处理32条文本，可根据内存调整
    results = []
    for i in tqdm(range(0, len(merged_df), batch_size), desc="情绪预测中"):
        batch_texts = merged_df["text"].iloc[i:i + batch_size].tolist()
        batch_results = sentiment_pipeline(batch_texts)
        results.extend(batch_results)

    # 5. 解析预测结果，生成sentiment_score和emotion_label
    emotion_df = pd.DataFrame(results)
    # 转换为情绪强度（0~1，越接近1越积极，越接近0越消极）
    emotion_df["sentiment_score"] = emotion_df.apply(
        lambda row: row["score"] if row["label"] == "LABEL_1" else 1 - row["score"],
        axis=1
    )
    # 分类情绪类型（中性：0.4~0.6；积极：>0.6；消极：<0.4，阈值可调整）
    emotion_df["emotion_label"] = pd.cut(
        emotion_df["sentiment_score"],
        bins=[-0.01, 0.4, 0.6, 1.01],
        labels=["消极", "中性", "积极"]
    )

    # 6. 合并到原始数据，保留核心字段
    final_df = pd.concat([
        merged_df.reset_index(drop=True),
        emotion_df[["sentiment_score", "emotion_label"]]
    ], axis=1)

    return final_df


def merge_engagement(emotion_df, dynamic_path):
    """
    关联视频互动量数据（engagement）
    :param emotion_df: 带情绪标签的DataFrame
    :param dynamic_path: 动态表路径（video_dynamic_2.csv）
    :return: 关联互动量后的DataFrame
    """
    # 读取动态表，提取每个视频的最新互动量（取最后一个状态）
    dynamic_df = pd.read_csv(dynamic_path, encoding="utf-8-sig")
    # 计算每个视频的历史状态长度，取最长的一条（最新状态）
    dynamic_df["历史状态长度"] = dynamic_df["历史状态"].apply(
        lambda x: len(eval(x)) if pd.notna(x) else 0
    )
    # 按url分组，取历史状态长度最大的行（最新互动数据）
    engagement_df = dynamic_df.loc[
        dynamic_df.groupby("url")["历史状态长度"].idxmax(),
        ["url", "当前点赞量", "当前播放量", "当前分享量"]
    ]
    # 重命名互动量字段，统一为engagement指标
    engagement_df.columns = ["url", "点赞量", "播放量", "分享量"]

    # 关联情绪数据和互动量（按url关联）
    # 注意：文本表中一个url可能对应多条记录（1标题+1描述+N评论），这里保留所有文本-情绪-互动量对应关系
    merged_emotion_engagement = pd.merge(
        emotion_df,
        engagement_df,
        on="url",
        how="inner"  # 只保留有互动数据的视频
    )
    print(f"关联互动量完成，共{len(merged_emotion_engagement)}条有效记录")
    return merged_emotion_engagement


def plot_emotion_engagement(merged_df, output_dir="情绪-互动量分析图/"):
    """
    绘制情绪类型与互动量的关系图
    :param merged_df: 关联情绪和互动量的DataFrame
    :param output_dir: 图表保存目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)  # 创建保存目录

    # 1. 按视频和情绪类型分组，计算平均互动量（避免单视频多条评论的重复影响）
    # 取每个视频的标题/描述情绪作为代表（评论情绪可单独分析）
    video_level = merged_df[merged_df["texttype"].isin(["title", "description"])].drop_duplicates("url")

    # 2. 绘制情绪类型与播放量的关系（箱线图，更能体现分布）
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="emotion_label",
        y="播放量",
        data=video_level,
        order=["积极", "中性", "消极"],
        palette=["#2ecc71", "#f39c12", "#e74c3c"]
    )
    plt.yscale("log")  # 播放量差异大，用对数刻度
    plt.title("不同情绪类型视频的播放量分布", fontsize=14)
    plt.xlabel("情绪类型", fontsize=12)
    plt.ylabel("播放量（对数刻度）", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/情绪类型-播放量分布.png", dpi=300)
    plt.close()

    # 3. 绘制情绪类型与点赞量的关系（柱状图，展示均值）
    plt.figure(figsize=(10, 6))
    emotion_avg_like = video_level.groupby("emotion_label")["点赞量"].mean().reindex(["积极", "中性", "消极"])
    emotion_avg_like.plot(kind="bar", color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.8)
    plt.title("不同情绪类型视频的平均点赞量", fontsize=14)
    plt.xlabel("情绪类型", fontsize=12)
    plt.ylabel("平均点赞量", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    # 添加数值标签
    for i, v in enumerate(emotion_avg_like):
        plt.text(i, v + 50, f"{int(v)}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/情绪类型-平均点赞量.png", dpi=300)
    plt.close()

    # 4. 统计各情绪类型占比（验证是否高比例情绪性内容）
    plt.figure(figsize=(8, 6))
    emotion_ratio = merged_df["emotion_label"].value_counts(normalize=True).reindex(["积极", "中性", "消极"]) * 100
    emotion_ratio.plot(kind="pie", autopct="%.1f%%", colors=["#2ecc71", "#f39c12", "#e74c3c"])
    plt.title("文本情绪类型占比分布", fontsize=14)
    plt.ylabel("")  # 去除y轴标签
    plt.tight_layout()
    plt.savefig(f"{output_dir}/情绪类型占比.png", dpi=300)
    plt.close()

    print(f"所有图表已保存至：{output_dir}")


# ------------------- 主函数：串联所有步骤 -------------------
def main():
    # 输入文件路径（请修改为你的实际路径）
    merged_text_path = "merged_text_data.csv"  # 合并文本表（带发布者）
    dynamic_path = "video_dynamic_2.csv"  # 动态表（含互动量）
    output_emotion_path = "带情绪标签的文本数据.csv"  # 输出带情绪标签的数据表

    # 步骤1：预测情绪
    emotion_df = predict_emotion(merged_text_path)
    # 步骤2：关联互动量
    merged_emotion_engagement = merge_engagement(emotion_df, dynamic_path)
    # 步骤3：保存带情绪标签的数据表
    merged_emotion_engagement.to_csv(output_emotion_path, index=False, encoding="utf-8-sig")
    print(f"带情绪标签的数据表已保存至：{output_emotion_path}")

    # 步骤4：绘制情绪与互动量关系图
    plot_emotion_engagement(merged_emotion_engagement)

    # 输出关键统计结果（验证高比例情绪性内容）
    print("\n=== 情绪类型分布统计 ===")
    emotion_count = merged_emotion_engagement["emotion_label"].value_counts()
    emotion_ratio = merged_emotion_engagement["emotion_label"].value_counts(normalize=True) * 100
    stats = pd.DataFrame({"数量": emotion_count, "占比(%)": emotion_ratio.round(2)})
    print(stats)
    print("\n结论：" + ("存在高比例情绪性内容（积极+消极占比>50%）"
                       if (stats.loc["积极", "占比(%)"] + stats.loc["消极", "占比(%)"]) > 50
                       else "情绪性内容占比不高（积极+消极占比≤50%）"))


if __name__ == "__main__":
    main()