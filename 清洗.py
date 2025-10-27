import pandas as pd
import numpy as np

def process_data(static_path, dynamic_path, comment_path,
                 processed_static, processed_dynamic, processed_comment):
    """
    观察数据情况后完整数据清洗流程：
    1. 剔除最后状态评论量为0的视频（静态表+动态表同步）
    2. 填充评论表空IP地址为“未知”
    3. 标记动态表中“播放量异常缺失”（互动量>0但播放量=0）
    """
    # 1. 读取数据
    static_df = pd.read_csv(static_path, encoding="utf-8-sig")
    dynamic_df = pd.read_csv(dynamic_path, encoding="utf-8-sig")
    comment_df = pd.read_csv(comment_path, encoding="utf-8-sig")

    # ---------------------- 步骤1：筛选保留最后状态评论量>0的视频 ----------------------
    # 1.1 标记每个记录的历史状态长度
    dynamic_df["历史状态长度"] = dynamic_df["历史状态"].apply(
        lambda x: len(eval(x)) if pd.notna(x) else 0
    )

    # 1.2 找到每个视频的最后一个状态
    max_length_per_url = dynamic_df.groupby("url")["历史状态长度"].max().reset_index()
    max_length_per_url.columns = ["url", "最大历史长度"]

    last_state_df = pd.merge(dynamic_df, max_length_per_url, on="url", how="inner")
    last_state_df = last_state_df[last_state_df["历史状态长度"] == last_state_df["最大历史长度"]]

    # 1.3 确定需要保留的视频URL（最后状态评论量>0）
    keep_urls = last_state_df[last_state_df["当前评论量"] > 0]["url"].unique()
    print(f"总视频数：{dynamic_df['url'].nunique()} | 保留视频数（最后状态评论>0）：{len(keep_urls)}")

    # ---------------------- 步骤2：同步过滤静态表和动态表 ----------------------
    # 2.1 过滤静态表
    static_filtered = static_df[static_df["url"].isin(keep_urls)].copy()

    # 2.2 过滤动态表（保留符合条件的URL）
    dynamic_filtered = dynamic_df[dynamic_df["url"].isin(keep_urls)].copy()
    dynamic_filtered = dynamic_filtered.drop(columns=["历史状态长度", "最大历史长度"], errors="ignore")

    # ---------------------- 步骤3：标记动态表中播放量异常缺失 ----------------------
    # 条件：播放量=0 且 （点赞量>0 或 分享量>0 或 收藏量>0）→ 标记为缺失（NaN）
    missing_condition = (dynamic_filtered["当前播放量"] == 0) & (
        (dynamic_filtered["当前点赞量"] > 0) |
        (dynamic_filtered["当前分享量"] > 0) |
        (dynamic_filtered["当前收藏量"] > 0)
    )
    # 标记播放量为缺失值
    dynamic_filtered.loc[missing_condition, "当前播放量"] = np.nan
    # 新增标记列（1=缺失，0=正常）
    dynamic_filtered["播放量是否缺失"] = missing_condition.astype(int)
    print(f"动态表中标记为播放量缺失的记录数：{missing_condition.sum()}条")

    # ---------------------- 步骤4：处理评论表 ----------------------
    # 4.1 填充空IP地址为“未知”
    comment_df["IP地址"] = comment_df["IP地址"].replace(
        to_replace=[np.nan, "", " "],
        value="未知"
    )
    # 4.2 过滤评论表（仅保留关联视频）
    comment_filtered = comment_df[comment_df["url"].isin(keep_urls)].copy()

    # ---------------------- 保存结果 ----------------------
    static_filtered.to_csv(processed_static, index=False, encoding="utf-8-sig")
    dynamic_filtered.to_csv(processed_dynamic, index=False, encoding="utf-8-sig")
    comment_filtered.to_csv(processed_comment, index=False, encoding="utf-8-sig")

    print(f"\n数据处理完成！")
    print(f"静态表：{len(static_filtered)}条（原{len(static_df)}条）")
    print(f"动态表：{len(dynamic_filtered)}条（原{len(dynamic_df)}条）")
    print(f"评论表：{len(comment_filtered)}条（原{len(comment_df)}条）")


# ------------------- 代码调用 -------------------
if __name__ == "__main__":
    # 输入文件路径
    STATIC_PATH = "video_static.csv"
    DYNAMIC_PATH = "video_dynamic.csv"
    COMMENT_PATH = "video_comments_deduped.csv"

    # 输出文件路径（可根据需要修改）
    PROCESSED_STATIC = "video_static_2.csv"
    PROCESSED_DYNAMIC = "video_dynamic_2.csv"
    PROCESSED_COMMENT = "video_comments_deduped.csv"  # 评论表保持原文件名

    # 执行完整处理流程
    process_data(
        STATIC_PATH, DYNAMIC_PATH, COMMENT_PATH,
        PROCESSED_STATIC, PROCESSED_DYNAMIC, PROCESSED_COMMENT
    )