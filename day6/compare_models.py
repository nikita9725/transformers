from pathlib import Path

import joblib
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from common import (
    SentimentDataset,
    evaluate,
    get_embeddings,
    get_tokenizer,
    load_sentiment_split,
)

# День 6, задача 1: сравнение baseline (день 4) и fine-tuned (день 5)

# Загрузка fine-tuned модели из дня 5
MODEL_FT_DIR = Path(__file__).parent.parent / "models" / "fine_tuned_model"
model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_FT_DIR)
tokenizer_ft = AutoTokenizer.from_pretrained(MODEL_FT_DIR)
model_ft.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)

# Подготовка данных (тот же сплит, что в дне 4)
train_texts, val_texts, train_labels, val_labels = load_sentiment_split()

# Загрузка baseline модели из дня 4
print("Загрузка baseline модели (день 4)...")
MODEL_PATH = Path(__file__).parent.parent / "day4" / "baseline_model.pkl"
clf = joblib.load(MODEL_PATH)

# Получение эмбеддингов для baseline
tokenizer_base = get_tokenizer("distilbert-base-uncased")
model_base = AutoModel.from_pretrained("distilbert-base-uncased")
model_base.eval()

X_val = get_embeddings(val_texts, tokenizer_base, model_base)
y_pred_bl = clf.predict(X_val)
acc_bl = accuracy_score(val_labels, y_pred_bl)
f1_bl = f1_score(val_labels, y_pred_bl, average="macro")

# Оценка fine-tuned модели
print("Оценка fine-tuned модели (день 5)...")
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer_ft, max_length=64)
val_loader = DataLoader(val_dataset, batch_size=16)
acc_ft, f1_ft = evaluate(model_ft, val_loader, device)

# Сравнение
print("\nСравнение моделей:")
print(f"Baseline (день 4, логрегрессия): accuracy {acc_bl:.4f} | macro F1 {f1_bl:.4f}")
print(f"Fine-tuned (день 5): accuracy {acc_ft:.4f} | macro F1 {f1_ft:.4f}")
print(f"\nПрирост: accuracy +{acc_ft - acc_bl:.4f} | macro F1 +{f1_ft - f1_bl:.4f}")
