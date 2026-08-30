from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from common import (
    load_baseline_model,
    load_fine_tuned_model,
    load_sentiment_split,
)
from day6.predict import predict_baseline, predict_fine_tuned

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

# Метрики для fine-tuned модели
print("\n" + "=" * 80)
print("=== Fine-tuned Model ===")
print("=" * 80)
print(classification_report(val_labels, y_pred_ft, target_names=["negative", "positive"]))
f1_ft = f1_score(val_labels, y_pred_ft, average="macro")
acc_ft = accuracy_score(val_labels, y_pred_ft)

# Метрики для baseline модели
print("=" * 80)
print("=== Baseline Model ===")
print("=" * 80)
print(classification_report(val_labels, y_pred_bl, target_names=["negative", "positive"]))
f1_bl = f1_score(val_labels, y_pred_bl, average="macro")
acc_bl = accuracy_score(val_labels, y_pred_bl)

# Сравнение
print("=" * 80)
print("=== Сравнение ===")
print("=" * 80)
print(f"Fine-tuned - F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
print(f"Baseline   - F1: {f1_bl:.4f}, Accuracy: {acc_bl:.4f}")
print(f"\nУлучшение F1: {(f1_ft - f1_bl) / f1_bl * 100:.2f}%")
print(f"Улучшение Accuracy: {(acc_ft - acc_bl) / acc_bl * 100:.2f}%")
print(f"\nАбсолютное улучшение F1: +{f1_ft - f1_bl:.4f}")
print(f"Абсолютное улучшение Accuracy: +{acc_ft - acc_bl:.4f}")

# Сохранение результатов в файл
results_path = Path(__file__).parent.parent / "comparison_results.txt"
with open(results_path, "w") as f:
    f.write("=== Сравнение моделей ===\n\n")
    f.write("Fine-tuned Model:\n")
    f.write(f"  F1 (macro): {f1_ft:.4f}\n")
    f.write(f"  Accuracy: {acc_ft:.4f}\n")
    f.write("\nBaseline Model:\n")
    f.write(f"  F1 (macro): {f1_bl:.4f}\n")
    f.write(f"  Accuracy: {acc_bl:.4f}\n")
    f.write(f"\nУлучшение F1: {(f1_ft - f1_bl) / f1_bl * 100:.2f}%\n")
    f.write(f"Улучшение Accuracy: {(acc_ft - acc_bl) / acc_bl * 100:.2f}%\n")

print(f"\nРезультаты сохранены: {results_path}")
