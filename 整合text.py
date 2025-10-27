import pandas as pd
import numpy as np


def merge_text_data(static_path, comment_path, output_path):
    """
    合并视频静态表和评论表，生成包含url、text、texttype、发布者的统一文本表
    :param static_path: 视频静态表路径（video_static_2.csv）
    :param comment_path: 评论表路径（video_comments_deduped.csv）
    :param output_path: 输出统一文本表的路径
    """
    # 1. 读取两个原始表
    static_df = pd.read_csv(static_path, encoding="utf-8-sig")
    comment_df = pd.read_csv(comment_path, encoding="utf-8-sig")

    # ---------------------- 2. 处理视频静态表：提取标题和描述（含发布用户ID） ----------------------
    # 提取“标题”→ texttype=title，发布者=发布用户ID
    title_data = static_df[["url", "标题", "发布用户ID"]].copy()
    title_data.columns = ["url", "text", "发布者"]  # 重命名为统一字段
    title_data["texttype"] = "title"  # 标记文本类型为“标题”

    # 提取“描述”→ texttype=description，发布者=发布用户ID
    desc_data = static_df[["url", "描述", "发布用户ID"]].copy()
    desc_data.columns = ["url", "text", "发布者"]  # 重命名为统一字段
    desc_data["texttype"] = "description"  # 标记文本类型为“描述”

    # ---------------------- 3. 处理评论表：提取评论内容（含评论用户ID） ----------------------
    comment_text_data = comment_df[["url", "评论内容", "评论用户ID"]].copy()
    comment_text_data.columns = ["url", "text", "发布者"]  # 重命名为统一字段（评论用户ID→发布者）
    comment_text_data["texttype"] = "comment"  # 标记文本类型为“评论”

    # ---------------------- 4. 合并所有文本数据并清洗 ----------------------
    # 合并标题、描述、评论数据
    merged_df = pd.concat(
        [title_data, desc_data, comment_text_data],
        ignore_index=True  # 重置索引，避免重复
    )

    # 清洗空文本：过滤text为空字符串、NaN、纯空格的记录
    merged_df["text"] = merged_df["text"].replace([np.nan, "", " "], None)
    merged_df = merged_df.dropna(subset=["text"])  # 删除text为空的行
    merged_df = merged_df[merged_df["text"].str.strip() != ""]  # 再过滤纯空格文本

    # 确保“发布者”列无缺失（若有缺失用“未知”填充）
    merged_df["发布者"] = merged_df["发布者"].fillna("未知")

    # ---------------------- 5. 保存最终统一文本表 ----------------------
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 输出统计信息，验证结果
    print(f"数据合并完成！")
    print(f"原始静态表记录数：{len(static_df)}（每个视频1条标题+1条描述，共{len(static_df) * 2}条文本）")
    print(f"原始评论表记录数：{len(comment_df)}（每条评论1条文本）")
    print(f"清洗后统一文本表记录数：{len(merged_df)}")
    print(f"各文本类型分布：")
    print(merged_df["texttype"].value_counts())
    print(f"\n最终表字段：{merged_df.columns.tolist()}")
    print(f"输出文件路径：{output_path}")


# ------------------- 代码调用（请修改为你的实际文件路径） -------------------
if __name__ == "__main__":
    # 输入文件路径（你的静态表和评论表）
    STATIC_PATH = "video_static_2.csv"  # 视频静态表
    COMMENT_PATH = "video_comments_deduped.csv"  # 去重后的评论表

    # 输出文件路径（统一文本表，后续用于情绪识别）
    OUTPUT_PATH = "merged_text_data.csv"  # 新增“发布者”后文件名可区分

    # 执行合并
    merge_text_data(STATIC_PATH, COMMENT_PATH, OUTPUT_PATH)