import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from common import (
    MAX_LENGTH,
    SentimentDataset,
    evaluate,
    get_embeddings,
    load_baseline_model,
    load_fine_tuned_model,
    load_sentiment_split,
)

# День 6, задача 1: сравнение baseline (день 4) и fine-tuned (день 5)

# Загрузка fine-tuned модели из дня 5
model_ft, tokenizer_ft = load_fine_tuned_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)  # type: ignore[arg-type]

# Подготовка данных (тот же сплит, что в дне 4)
train_texts, val_texts, train_labels, val_labels = load_sentiment_split()

# Загрузка baseline модели из дня 4
print("Загрузка baseline модели (день 4)...")
clf, tokenizer_base, model_base = load_baseline_model()

# Получение эмбеддингов для baseline
X_val = get_embeddings(val_texts, tokenizer_base, model_base)
y_pred_bl = clf.predict(X_val)
acc_bl = accuracy_score(val_labels, y_pred_bl)
f1_bl = f1_score(val_labels, y_pred_bl, average="macro")

# Оценка fine-tuned модели
print("Оценка fine-tuned модели (день 5)...")
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer_ft, max_length=MAX_LENGTH)
val_loader = DataLoader(val_dataset, batch_size=16)
acc_ft, f1_ft = evaluate(model_ft, val_loader, device)

# Сравнение
print("\nСравнение моделей:")
print(f"Baseline (день 4, логрегрессия): accuracy {acc_bl:.4f} | macro F1 {f1_bl:.4f}")
print(f"Fine-tuned (день 5): accuracy {acc_ft:.4f} | macro F1 {f1_ft:.4f}")
print(f"\nПрирост: accuracy +{acc_ft - acc_bl:.4f} | macro F1 +{f1_ft - f1_bl:.4f}")
