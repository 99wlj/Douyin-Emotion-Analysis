import pandas as pd
import matplotlib.pyplot as plt

model_files = {
    "XGBoost": "processed_video_data.csv",
    "Random Forest": "processed_video_data_RF.csv",
    "LightGBM": "processed_video_data_LGBM.csv"
}

metrics_data = [
    {"Model": "XGBoost", "Accuracy": 0.9997, "F1": 0.9997, "AUC": 1.0000},
    {"Model": "Random Forest", "Accuracy": 0.5103, "F1": 0.4715, "AUC": 0.5158},
    {"Model": "LightGBM", "Accuracy": 0.9998, "F1": 0.9998, "AUC": 1.0000}
]

df_metrics = pd.DataFrame(metrics_data)
print(df_metrics)

# 保存表格
df_metrics.to_csv("model_performance_summary.csv", index=False, encoding="utf-8-sig")

# 可视化
plt.figure(figsize=(8, 5))
for metric in ["Accuracy", "F1", "AUC"]:
    plt.plot(df_metrics["Model"], df_metrics[metric], marker="o", label=metric)

plt.ylim(0.99, 1.001)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

