import os
import argparse
import warnings
import traceback  # 新增：用于打印详细报错
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

# 设置 matplotlib 后端，防止无界面环境下报错
import matplotlib
matplotlib.use('Agg') 

def build_label(df: pd.DataFrame, quantile: float = 0.7) -> pd.Series:
    for col in ["点赞量", "分享量"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    metric = 0.6 * np.log1p(df["点赞量"]) + 0.4 * np.log1p(df["分享量"])
    threshold = metric.quantile(quantile)
    y = (metric >= threshold).astype(int)
    return y


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score"), errors="coerce").fillna(0)
    df["text"] = df.get("text").astype(str)
    df["text_len"] = df["text"].apply(lambda s: len(s) if isinstance(s, str) else 0)

    df["emotion_label"] = df.get("emotion_label").astype(str).fillna("")
    df["texttype"] = df.get("texttype").astype(str).fillna("")

    feature_cols = [
        "sentiment_score",
        "text_len",
        "emotion_label",
        "texttype",
    ]
    return df[feature_cols]


def get_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            # tree_method="hist", # 可选，视版本而定
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1 # 减少日志
        ),
    }
    return models


def build_pipeline(model):
    categorical_cols = ["emotion_label", "texttype"]
    numeric_cols = ["sentiment_score", "text_len"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipe


def proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if prob.shape[1] == 2:
            return prob[:, 1]
        else:
            return prob.max(axis=1)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
        if score.ndim == 1:
            return (score - score.min()) / (score.max() - score.min() + 1e-9)
        else:
            return score.max(axis=1)
    else:
        return model.predict(X)


def run(data_path: str, quantile: float, test_size: float, max_rows: int, do_shap: bool):
    print(f"加载数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"错误: 文件不存在 {data_path}")
        return

    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)

    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    y = build_label(df, quantile=quantile)
    X = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = get_models()
    metrics_rows = []

    best_auc = -1.0
    best_name = None
    best_fitted = None
    best_X_train = None

    print("开始训练模型...")
    for name, model in models.items():
        pipe = build_pipeline(model)
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(f"模型 {name} 训练失败: {e}")
            continue

        y_pred = pipe.predict(X_test)
        y_score = proba_or_score(pipe.named_steps["model"], pipe.named_steps["preprocessor"].transform(X_test))

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_score)
        except Exception:
            auc = np.nan

        metrics_rows.append({"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc})
        print(f"[{name}] Accuracy={acc:.4f} F1={f1:.4f} AUC={auc if not np.isnan(auc) else float('nan'):.4f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_name = name
            best_fitted = pipe
            best_X_train = X_train

    if not metrics_rows:
        print("没有模型训练成功。")
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="AUC", ascending=False, na_position="last")
    print("\n模型指标表：")
    print(metrics_df.to_string(index=False))

    # SHAP 部分
    if do_shap and best_fitted is not None:
        print(f"\n生成 SHAP 图（最佳模型：{best_name}）")
        
        # 1. 准备解释数据
        expl_rows = min(2000, len(best_X_train)) # 稍微减少采样数以加快速度
        X_expl = best_X_train.sample(n=expl_rows, random_state=42)

        preproc = best_fitted.named_steps["preprocessor"]
        X_expl_proc = preproc.transform(X_expl)

        # 2. 准备特征名称
        numeric_cols = ["sentiment_score", "text_len"]
        try:
            cat_ohe = preproc.named_transformers_["cat"]
            if hasattr(cat_ohe, "get_feature_names_out"):
                cat_feature_names = list(cat_ohe.get_feature_names_out(["emotion_label", "texttype"]))
            else:
                cat_feature_names = list(cat_ohe.get_feature_names(["emotion_label", "texttype"])) # 旧版sklearn兼容
            feature_names = numeric_cols + cat_feature_names
        except Exception as e:
            print(f"特征名获取失败，将使用索引代替: {e}")
            feature_names = None

        model = best_fitted.named_steps["model"]

        try:
            # 3. 计算 SHAP 值
            # 注意：传入 model 而不是 pipe
            explainer = shap.TreeExplainer(model)
            
            # 关键修复1: check_additivity=False 防止报错
            print("正在计算 SHAP values...")
            shap_values = explainer.shap_values(X_expl_proc, check_additivity=False)

            # 关键修复2: 处理不同模型返回的 shap_values 格式差异
            shap_vals_for_plot = None
            
            # 情况A: 如果是 list (RandomForest 通常返回 [class0_matrix, class1_matrix])
            if isinstance(shap_values, list):
                print(f"检测到 SHAP 返回 list，长度: {len(shap_values)}")
                if len(shap_values) >= 2:
                    shap_vals_for_plot = shap_values[1] # 取正类
                else:
                    shap_vals_for_plot = shap_values[0]
            
            # 情况B: 已经是数组 (XGBoost/LGBM 二分类通常直接返回 (N, features))
            else:
                print(f"检测到 SHAP 返回 array，形状: {np.shape(shap_values)}")
                shap_vals_for_plot = shap_values

            # 4. 绘图
            if shap_vals_for_plot is not None:
                # Bar Plot
                plt.figure(figsize=(10, 8), dpi=100)
                shap.summary_plot(shap_vals_for_plot, X_expl_proc, feature_names=feature_names, plot_type="bar", show=False)
                plt.tight_layout()
                bar_path = os.path.abspath("shap_bar.png")
                plt.savefig(bar_path, bbox_inches='tight')
                plt.close() # 必须关闭

                # Summary Plot (Beeswarm)
                plt.figure(figsize=(10, 8), dpi=100)
                shap.summary_plot(shap_vals_for_plot, X_expl_proc, feature_names=feature_names, show=False)
                plt.tight_layout()
                summary_path = os.path.abspath("shap_summary.png")
                plt.savefig(summary_path, bbox_inches='tight')
                plt.close()

                print(f"SHAP 图生成成功！\n保存路径:\n1. {bar_path}\n2. {summary_path}")
            else:
                print("错误: 无法解析 shap_values，绘图跳过。")

        except Exception as e:
            print("\n============ SHAP 生成失败 ============")
            print("错误详情:")
            traceback.print_exc() # 打印完整堆栈信息
            print("建议检查: 模型版本与SHAP库版本的兼容性，或数据是否存在极值。")


def parse_args():
    parser = argparse.ArgumentParser()
    # 默认路径修改为当前目录下的示例，方便测试，请根据实际情况调整
    parser.add_argument("--data", type=str, default="data.csv", help="数据文件路径")
    parser.add_argument("--quantile", type=float, default=0.7)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--shap", action="store_true", default=True) # 默认开启方便调试
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 如果没有 args.data 文件，生成一个假的测试
    if not os.path.exists(args.data) and args.data == "data.csv":
        print("未找到数据，生成模拟数据用于测试代码逻辑...")
        dummy_df = pd.DataFrame({
            "text": ["abc"] * 100,
            "sentiment_score": np.random.rand(100),
            "emotion_label": np.random.choice(["happy", "sad"], 100),
            "texttype": np.random.choice(["type1", "type2"], 100),
            "点赞量": np.random.randint(0, 100, 100),
            "分享量": np.random.randint(0, 100, 100)
        })
        dummy_df.to_csv("data.csv", index=False)
    
    run(
        data_path=args.data,
        quantile=args.quantile,
        test_size=args.test_size,
        max_rows=args.max_rows,
        do_shap=args.shap,
    )