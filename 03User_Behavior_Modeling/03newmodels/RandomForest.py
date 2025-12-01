import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
import shap
import warnings

warnings.filterwarnings('ignore')

file_path = r"D:\Program Files\VscodeProjects\社会网络分析\data\processed\StructBERTDataSet.csv"

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='gbk')

df['点赞量'] = df['点赞量'].fillna(0)
df['分享量'] = df['分享量'].fillna(0)
df['播放量'] = df['播放量'].fillna(0)

df['total_interactions'] = df['点赞量'] + df['分享量']

df['text_length'] = df['text'].astype(str).apply(len)

emotion_map = {'消极': 0, '中性': 1, '积极': 2}
df['emotion_encoded'] = df['emotion_label'].map(emotion_map)

df['platform'] = df['url'].apply(lambda x: 'douyin' if 'douyin' in x else ('kuaishou' if 'kuaishou' in x else 'other'))
platform_dummies = pd.get_dummies(df['platform'], prefix='platform')

median_engagement = df['total_interactions'].median()
df['high_engagement'] = (df['total_interactions'] > median_engagement).astype(int)

print(f"Engagement Median Split Point: {median_engagement}")

features = ['sentiment_score', 'text_length', 'emotion_encoded']
X = pd.concat([df[features], platform_dummies], axis=1)
y = df['high_engagement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\n--- Model Performance Report ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

try:
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
except:
    print("AUC Score: Not defined (likely single class in test set)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


try:
    print("\nCalculating SHAP values (using a sample of 2000 records for speed)...")
    if len(X_test) > 2000:
        X_shap_sample = X_test.sample(n=2000, random_state=42)
    else:
        X_shap_sample = X_test

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap_sample, check_additivity=False)
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
    else:
        shap_vals_class1 = shap_values

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['sentiment_score'], df['total_interactions'], 
                c=df['high_engagement'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='High Engagement (1=Yes)')
    plt.xlabel('Sentiment Score (Raw)')
    plt.ylabel('Total Interactions (Likes + Shares)')
    plt.title('Sentiment Score vs. User Engagement')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('sentiment_vs_engagement.png')
    print("Saved plot: sentiment_vs_engagement.png")


    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_class1, X_shap_sample, show=False)
    plt.title("SHAP Feature Importance (Predicting High Engagement)")
    plt.savefig('shap_summary.png', bbox_inches='tight')
    print("Saved plot: shap_summary.png")

except Exception as e:
    print(f"Could not generate SHAP plot: {e}")


performance_data = {
    'Metric': ['Accuracy', 'Recall', 'F1_Score', 'ROC_AUC', 'Engagement_Threshold'],
    'Value': [
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred), # 新增
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        median_engagement
    ]
}
pd.DataFrame(performance_data).to_csv('model_performance_summary.csv', index=False)
print("Saved CSV: model_performance_summary.csv")

# 8. 简单的统计验证：高分情绪是否互动更高？
high_sentiment_avg = df[df['sentiment_score'] > 0.6]['total_interactions'].mean()
low_sentiment_avg = df[df['sentiment_score'] < 0.4]['total_interactions'].mean()
print(f"\n--- Statistical Check ---")
print(f"Avg Interactions (High Sentiment > 0.6): {high_sentiment_avg:.2f}")
print(f"Avg Interactions (Low Sentiment < 0.4): {low_sentiment_avg:.2f}")