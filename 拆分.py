import json
import pandas as pd
from typing import Dict, List


def split_video_data(raw_data_path: str, static_save_path: str, dynamic_save_path: str, comment_save_path: str):
    """
    拆分视频数据为静态表、动态指标表和评论表（评论表按用户ID+内容+时间去重）
    :param raw_data_path: 原始数据JSON文件路径
    :param static_save_path: 静态表保存路径（如"static_data.csv"）
    :param dynamic_save_path: 动态指标表保存路径（如"dynamic_data.csv"）
    :param comment_save_path: 评论表保存路径（如"comment_data.csv"）
    """
    # 1. 读取原始数据
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. 定义字段列表
    static_fields = [
        "url", "描述", "标题", "发布用户ID", "视频时长（秒）",
        "平台", "发布时间", "话题", "影响力等级"
    ]
    dynamic_fields = [
        "url", "当前时间", "当前粉丝量", "当前点赞量", "当前播放量",
        "当前分享量", "当前收藏量", "当前评论量", "历史状态"
    ]
    comment_fields = [
        "url", "评论用户ID", "评论内容", "评论时间", "IP地址"
    ]

    # 3. 初始化数据列表
    static_data: List[Dict] = []
    dynamic_data: List[Dict] = []
    comment_data: List[Dict] = []
    static_urls: set = set()

    # 4. 遍历原始数据拆分
    for record_id, record in raw_data.items():
        current_url = record["url"]

        # 4.1 处理静态表（去重）
        if record["历史状态"] == [] and current_url not in static_urls:
            static_record = {field: record.get(field, "") for field in static_fields}
            static_data.append(static_record)
            static_urls.add(current_url)

        # 4.2 处理动态指标表
        dynamic_record = {field: record.get(field, "") for field in dynamic_fields}
        dynamic_data.append(dynamic_record)

        # 4.3 处理评论表（先收集所有评论，后续统一去重）
        comments = record.get("评论", {})
        for comment_id, comment in comments.items():
            comment_record = {
                "url": current_url,
                "评论用户ID": comment.get("评论用户ID", ""),
                "评论内容": comment.get("评论内容", ""),
                "评论时间": comment.get("评论时间", ""),
                "IP地址": comment.get("IP地址", "")
            }
            comment_data.append(comment_record)

    # 5. 数据保存
    # 静态表
    static_df = pd.DataFrame(static_data)
    static_df.to_csv(static_save_path, index=False, encoding="utf-8-sig")

    # 动态指标表
    dynamic_df = pd.DataFrame(dynamic_data)
    dynamic_df.to_csv(dynamic_save_path, index=False, encoding="utf-8-sig")

    # 评论表（关键：按“评论用户ID+评论内容+评论时间”联合去重）
    comment_df = pd.DataFrame(comment_data)
    if not comment_df.empty:
        # 定义去重的联合唯一键
        duplicate_keys = ["评论用户ID", "评论内容", "评论时间"]
        # 去重：保留重复项中的第一条，重置索引作为评论编号
        comment_df_deduped = comment_df.drop_duplicates(
            subset=duplicate_keys,
            keep="first"  # 保留第一次出现的重复评论
        ).reset_index(drop=True)
        # 新增“评论编号”字段（可选，若需要明确编号）
        comment_df_deduped.insert(0, "评论编号", comment_df_deduped.index + 1)
    else:
        # 若没有评论数据，创建空表并保留字段
        comment_df_deduped = pd.DataFrame(columns=["评论编号"] + comment_fields)

    # 保存去重后的评论表
    comment_df_deduped.to_csv(comment_save_path, index=False, encoding="utf-8-sig")

    # 输出拆分结果
    print(f"数据拆分完成！")
    print(f"静态表（{len(static_df)}条）已保存至：{static_save_path}")
    print(f"动态指标表（{len(dynamic_df)}条）已保存至：{dynamic_save_path}")
    print(f"评论表（去重前{len(comment_df)}条 → 去重后{len(comment_df_deduped)}条）已保存至：{comment_save_path}")


# ------------------- 代码调用示例 -------------------
if __name__ == "__main__":
    RAW_DATA_PATH = "released_dataset.json" 
    STATIC_SAVE_PATH = "video_static.csv"
    DYNAMIC_SAVE_PATH = "video_dynamic.csv"
    COMMENT_SAVE_PATH = "video_comments_deduped.csv"  # 去重后的评论表

    split_video_data(RAW_DATA_PATH, STATIC_SAVE_PATH, DYNAMIC_SAVE_PATH, COMMENT_SAVE_PATH)