import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings

# ---------------------------------------------------------
# 1. 基础设置
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 2. 读取数据
# ---------------------------------------------------------
file_path = r"D:\Program Files\VscodeProjects\社会网络分析\data\processed\StructBERTDataSet.csv"
print(">>> 正在读取数据...")
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='gbk')

# ---------------------------------------------------------
# 3. 特征工程 (保持完整)
# ---------------------------------------------------------
df['total_interactions'] = df['点赞量'].fillna(0) + df['分享量'].fillna(0)
df['text'] = df['text'].astype(str)

df['has_brackets'] = (df['sentiment_score'] - 0.5).abs() * 2
df['text_length'] = df['text'].apply(len)
df['exclaim_count'] = df['text'].apply(lambda x: x.count('!') + x.count('！'))
df['question_count'] = df['text'].apply(lambda x: x.count('?') + x.count('？'))
df['sentiment_intensity'] = df['text'].apply(lambda x: 1 if ('【' in x or '[' in x) else 0)

features = [
    'sentiment_intensity', 
    'sentiment_score', 
    'text_length', 
    'exclaim_count', 
    'question_count', 
    'has_brackets'
]

# ---------------------------------------------------------
# 4. 训练模型
# ---------------------------------------------------------
X = df[features]
y = (df['total_interactions'] > df['total_interactions'].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. 计算 SHAP 值
# ---------------------------------------------------------
print(">>> 正在计算 SHAP 值...")
X_shap = X_test.sample(2000, random_state=42) if len(X_test) > 2000 else X_test
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_shap)

# 提取 Class 1
if isinstance(shap_values, list):
    vals = shap_values[1]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    vals = shap_values[:, :, 1]
else:
    vals = shap_values

# ---------------------------------------------------------
# 6. 绘图
# ---------------------------------------------------------
fig = plt.figure(figsize=(25, 6), dpi=300) 

shap.summary_plot(
    vals, 
    X_shap, 
    feature_names=features,
    plot_type="dot", 
    max_display=10, 
    alpha=0.6, 
    show=False 
)

# 确保不切数据，让它自动铺满这个超宽的画布
plt.autoscale(enable=True, axis='x', tight=True)

plt.title('Feature influence distribution', fontsize=20, pad=20)
plt.xlabel('SHAP Value', fontsize=14)
# 加大刻度字体，配合大图
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

save_name = 'SHAP.png'
plt.tight_layout()
plt.savefig(save_name, bbox_inches='tight', dpi=300)