from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from common import (
    load_baseline_model,
    load_fine_tuned_model,
    load_sentiment_split,
    predict_baseline,
    predict_fine_tuned,
)

# Директория для сохранения артефактов
OUTPUT_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка тестовых данных
print("Загрузка тестовых данных...")
train_texts, val_texts, train_labels, val_labels = load_sentiment_split()
print(f"Тестовых примеров: {len(val_texts)}")

# Загрузка fine-tuned модели
print("\nЗагрузка fine-tuned модели...")
model_ft, tokenizer_ft = load_fine_tuned_model()

# Загрузка baseline модели
print("Загрузка baseline модели...")
model_bl, tokenizer_bl, model_bl_emb = load_baseline_model()

# Получение предсказаний fine-tuned модели
print("\nПолучение предсказаний fine-tuned модели...")
preds_ft_all = predict_fine_tuned(val_texts, model_ft, tokenizer_ft)
y_pred_ft = [p["prediction"] for p in preds_ft_all]

# Получение предсказаний baseline модели
print("Получение предсказаний baseline модели...")
preds_bl_all = predict_baseline(val_texts, model_bl, tokenizer_bl, model_bl_emb)
y_pred_bl = [p["prediction"] for p in preds_bl_all]

# Построение confusion matrix
cm_ft = confusion_matrix(val_labels, y_pred_ft)
cm_bl = confusion_matrix(val_labels, y_pred_bl)

print("\nConfusion Matrix - Fine-tuned Model:")
print(cm_ft)
print("\nConfusion Matrix - Baseline Model:")
print(cm_bl)

# Визуализация — обе модели на одном figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Fine-tuned model
sns.heatmap(
    cm_ft,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
axes[0].set_title("Confusion Matrix - Fine-tuned Model", fontsize=14, fontweight="bold")
axes[0].set_ylabel("True Label", fontsize=12)
axes[0].set_xlabel("Predicted Label", fontsize=12)

# Baseline model
sns.heatmap(
    cm_bl,
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=axes[1],
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
axes[1].set_title("Confusion Matrix - Baseline Model", fontsize=14, fontweight="bold")
axes[1].set_ylabel("True Label", fontsize=12)
axes[1].set_xlabel("Predicted Label", fontsize=12)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png")
print(f"Сохранено: {OUTPUT_DIR / 'confusion_matrix.png'}")
plt.show()

# Статистика
print("\n" + "=" * 80)
print("Статистика:")
print(f"Fine-tuned - Правильных: {cm_ft[0, 0] + cm_ft[1, 1]}/{len(val_labels)}")
print(f"Baseline - Правильных: {cm_bl[0, 0] + cm_bl[1, 1]}/{len(val_labels)}")
print(f"Fine-tuned accuracy: {(cm_ft[0, 0] + cm_ft[1, 1]) / len(val_labels):.4f}")
print(f"Baseline accuracy: {(cm_bl[0, 0] + cm_bl[1, 1]) / len(val_labels):.4f}")
