import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# 1. 读取数据 ---------------------------------------------------
df = pd.read_csv("带情绪标签及互动量的文本数据.csv", dtype={"发布者": str}, low_memory=False)

# 重建情绪标签
def label_emotion(score):
    if score > 0.6:
        return "positive"
    elif score < 0.4:
        return "negative"
    else:
        return "neutral"
df["emotion_label"] = df["sentiment_score"].apply(label_emotion)

# 数值化互动数据
for col in ["点赞量", "播放量", "分享量"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
df = df[df["播放量"] > 0]

# 构建 engagement_score （目标变量）
df["like_rate"] = df["点赞量"] / df["播放量"]
df["share_rate"] = df["分享量"] / df["播放量"]
df["engagement_score"] = 0.6 * df["like_rate"] + 0.4 * df["share_rate"]
median_eng = df["engagement_score"].median()
df["engagement_label"] = (df["engagement_score"] > median_eng).astype(int)

# 2. 特征（仅情绪相关）----------------------------------------
scaler = MinMaxScaler()
df["sentiment_score_norm"] = scaler.fit_transform(df[["sentiment_score"]])

enc = OneHotEncoder(sparse_output=False)
emotion_encoded = enc.fit_transform(df[["emotion_label"]])
emotion_df = pd.DataFrame(emotion_encoded, columns=enc.get_feature_names_out(["emotion_label"]))
df = pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)

X = df[["sentiment_score_norm"] + list(emotion_df.columns)]
y = df["engagement_label"]

# 3. 训练模型 ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. 评估 --------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
print(classification_report(y_test, y_pred))

# 5. SHAP --------------------------------------------------------
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Feature Importance (Emotion-Only Random Forest)")
plt.tight_layout()
plt.show()