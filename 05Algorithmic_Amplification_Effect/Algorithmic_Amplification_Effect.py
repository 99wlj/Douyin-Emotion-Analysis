import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys
import os

# ================= 配置区域 (Configuration) =================
# 1. 输入文件路径 (使用了 r"" 防止转义错误)
input_file = r"D:\HuaweiMoveData\Users\32549\OneDrive\final_video_data_with_scores.csv" 

# 2. 输出文件夹路径
output_dir = r"05Algorithmic_Amplification_Effect\05result"

# 3. 日志文件路径
output_log_file = os.path.join(output_dir, "stage5_analysis_log.txt")
# ==========================================================

# --- 自动创建输出目录 ---
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"[系统] 已自动创建输出文件夹: {output_dir}")
    except Exception as e:
        print(f"[错误] 无法创建文件夹，请检查路径权限: {e}")
        sys.exit(1)

class Logger(object):
    """
    日志重定向类：将打印内容同时输出到控制台和TXT文件
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 启用日志记录
sys.stdout = Logger(output_log_file)

def run_stage5_analysis():
    print("="*60)
    print("           阶段5：算法放大效应分析 (完整数据导出版)           ")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. 数据读取与预处理
    # ---------------------------------------------------------
    if not os.path.exists(input_file):
        print(f"[错误] 找不到输入文件: {input_file}")
        return

    try:
        print("[1/4] 正在读取数据...")
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except UnicodeDecodeError:
            print("      (UTF-8解码失败，尝试GBK编码...)")
            df = pd.read_csv(input_file, encoding='gbk')
        
        print(f"      成功加载 {len(df)} 条记录。")

        # 检查并构建 'total_interactions'
        if 'total_interactions' not in df.columns:
            if '点赞量' in df.columns and '分享量' in df.columns:
                df['total_interactions'] = df['点赞量'].fillna(0) + df['分享量'].fillna(0)
                print("      已通过 (点赞+分享) 重新构建 total_interactions 列。")
            else:
                df['total_interactions'] = 0
                print("      [警告] 无法构建互动量，默认为0。")

        # 填充缺失值
        df['total_interactions'] = df['total_interactions'].fillna(0)
        df['sentiment_score'] = df['sentiment_score'].fillna(0.5) # 缺失默认为中性
        df['text_length'] = df['text_length'].fillna(0)
        if 'predicted_score' not in df.columns:
             # 如果上一阶段没跑成功，防止这里报错，生成一个假列供测试
             print("      [警告] 缺少 predicted_score 列，暂时用随机数填充以防代码崩溃（请检查上一阶段输出）。")
             df['predicted_score'] = np.random.rand(len(df))

        # --- 核心特征构造 ---
        # 1. 对数互动量 (Log Interactions): 解决数据长尾分布问题
        df['log_interactions'] = np.log1p(df['total_interactions'])
        
        # 2. 情绪强度 (Sentiment Intensity): 越接近0或1越强，0.5为中性
        # 计算公式：|分数 - 0.5|
        df['sentiment_intensity'] = (df['sentiment_score'] - 0.5).abs()
        
        print("[2/4] 数据预处理完成。已生成 'log_interactions' 和 'sentiment_intensity'。")

        # ---------------------------------------------------------
        # 2. 回归模型 A: 算法是否偏好情绪? (Predicted Score)
        # ---------------------------------------------------------
        print("\n[3/4] 正在执行回归分析...")
        print("\n--- 模型 A: 算法预测分回归 (验证算法机制) ---")
        
        # 公式：算法预测分 ~ 情绪得分 + 文本长度
        model_algo = ols('predicted_score ~ sentiment_score + text_length', data=df).fit()
        
        # 打印简要结果
        print(model_algo.summary().tables[1])
        
        # 导出详细结果到 CSV
        res_df_1 = model_algo.summary2().tables[1]
        file_res_1 = os.path.join(output_dir, "Regression_Algorithm_Preference.csv")
        res_df_1.to_csv(file_res_1)
        print(f"      >> [导出成功] 算法偏好回归表: {file_res_1}")

        # ---------------------------------------------------------
        # 3. 回归模型 B: 真实互动是否受情绪强度影响? (Real Interactions)
        # ---------------------------------------------------------
        print("\n--- 模型 B: 真实互动回归 (验证情绪极化/U型曲线) ---")
        
        # 公式：对数互动量 ~ 情绪强度 + 文本长度
        # 情绪强度显著说明：不管是极好还是极坏，只要情绪强，互动就高
        model_real = ols('log_interactions ~ sentiment_intensity + text_length', data=df).fit()
        
        # 打印简要结果
        print(model_real.summary().tables[1])
        
        # 导出详细结果到 CSV
        res_df_2 = model_real.summary2().tables[1]
        file_res_2 = os.path.join(output_dir, "Regression_Real_Amplification.csv")
        res_df_2.to_csv(file_res_2)
        print(f"      >> [导出成功] 真实传播放大回归表: {file_res_2}")

        # ---------------------------------------------------------
        # 4. 保存完整数据 (备份)
        # ---------------------------------------------------------
        print("\n[4/4] 正在保存完整过程数据...")
        full_data_path = os.path.join(output_dir, "stage5_full_data_processed.csv")
        df.to_csv(full_data_path, index=False, encoding='utf-8-sig')
        print(f"      >> [导出成功] 包含新变量的完整数据: {full_data_path}")

        # ---------------------------------------------------------
        # 5. 自动结论生成
        # ---------------------------------------------------------
        print("\n" + "="*30)
        print("       自动分析结论")
        print("="*30)
        
        # 提取模型B (情绪强度) 的P值和系数
        p_val = model_real.pvalues.get('sentiment_intensity', 1.0)
        coef = model_real.params.get('sentiment_intensity', 0.0)
        
        print(f"情绪强度(Intensity) 对 真实互动(Log Interactions) 的影响:")
        print(f"  - 系数 (Coef): {coef:.4f}")
        print(f"  - P值 (P>|t|): {p_val:.4f}")
        
        if p_val < 0.05 and coef > 0:
            print("\n✅ [结论] 数据支持‘算法放大效应’与‘情绪极化’假设。")
            print("   分析表明，情绪越激烈（偏离中性越远），视频获得的互动量显著越高。")
        elif p_val < 0.05 and coef < 0:
            print("\n❌ [结论] 负相关。情绪越激烈，互动量反而越低（可能受审核抑制）。")
        else:
            print("\n⚠️ [结论] 关系不显著。未发现情绪强度与互动量有明显的统计学关联。")

    except Exception as e:
        print(f"\n[严重错误] 分析过程中发生异常: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print(f"所有文件已保存至文件夹: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    run_stage5_analysis()