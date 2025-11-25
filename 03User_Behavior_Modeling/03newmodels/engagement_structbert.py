import os
import argparse
import warnings
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
    # 保持你原有的逻辑不变
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
            tree_method="hist",
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
            verbose=-1
        ),
    }
    return models


def build_pipeline(model):
    # 保持你原有的逻辑不变
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
    # 保持你原有的逻辑不变
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

    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)

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

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="AUC", ascending=False, na_position="last")
    print("\n模型指标表：")
    print(metrics_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 仅修改了以下 SHAP 绘图部分，修复了报错问题
    # -------------------------------------------------------------------------
    if do_shap and best_fitted is not None:
        print(f"\n生成 SHAP 图（最佳模型：{best_name}）")
        
        # 采样以加快速度
        expl_rows = min(2000, len(best_X_train))
        X_expl = best_X_train.sample(n=expl_rows, random_state=42)

        preproc = best_fitted.named_steps["preprocessor"]
        X_expl_proc = preproc.transform(X_expl)

        # 获取特征名 (兼容不同 sklearn 版本)
        numeric_cols = ["sentiment_score", "text_len"]
        cat_transformer = preproc.named_transformers_["cat"]
        if hasattr(cat_transformer, "get_feature_names_out"):
            cat_names = list(cat_transformer.get_feature_names_out(["emotion_label", "texttype"]))
        else:
            cat_names = list(cat_transformer.get_feature_names(["emotion_label", "texttype"]))
        feature_names = numeric_cols + cat_names

        model = best_fitted.named_steps["model"]
        try:
            explainer = shap.TreeExplainer(model)
            # 关键修改1: check_additivity=False 防止报错
            print("正在计算 SHAP values...")
            shap_values = explainer.shap_values(X_expl_proc, check_additivity=False)

            # 关键修改2: 处理不同模型(RF vs XGB)返回的格式差异
            # RandomForest 返回 list [Class0, Class1]，XGBoost 二分类返回 array
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_vals_for_plot = shap_values[1] # 取正类 (1)
                else:
                    shap_vals_for_plot = shap_values[0]
            else:
                shap_vals_for_plot = shap_values

            # 绘图 1: Bar
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals_for_plot, X_expl_proc, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            bar_path = os.path.abspath("shap_bar.png")
            plt.savefig(bar_path, bbox_inches='tight')
            plt.close()

            # 绘图 2: Summary
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals_for_plot, X_expl_proc, feature_names=feature_names, show=False)
            plt.tight_layout()
            summary_path = os.path.abspath("shap_summary.png")
            plt.savefig(summary_path, bbox_inches='tight')
            plt.close()

            print(f"SHAP 图已保存：\n{bar_path}\n{summary_path}")
        except Exception as e:
            # 打印详细错误方便调试
            import traceback
            traceback.print_exc()
            warnings.warn(f"SHAP 计算失败：{e}")


def parse_args():
    # 恢复了你的默认路径
    parser = argparse.ArgumentParser(description="StructBERT 数据集参与度预测与解释")
    parser.add_argument("--data", type=str, default=os.path.join("data", "processed", "StructBERTDataSet.csv"),
                        help="数据文件路径")
    parser.add_argument("--quantile", type=float, default=0.7, help="高参与度分位数阈值（0-1）")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集占比（0-1）")
    parser.add_argument("--max_rows", type=int, default=0, help="采样的最大行数，0 表示不采样")
    parser.add_argument("--shap", action="store_true", help="是否生成 SHAP 图")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data,
        quantile=args.quantile,
        test_size=args.test_size,
        max_rows=args.max_rows,
        do_shap=args.shap,
    )
