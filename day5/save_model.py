from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from common import (
    EN_MODEL,
    SentimentDataset,
    evaluate,
    get_tokenizer,
    load_sentiment_dataset,
    train_epoch,
)

# День 5, задача 8: сохранение модели
# Обучаем модель (те же 3 эпохи, что в задаче 7) и сохраняем через
# save_pretrained вместе с токенизатором: чекпоинт становится
# самодостаточным, в config.json сохраняется архитектура головы (num_labels)

NUM_EPOCHS = 3
MAX_LENGTH = 64
# Всё, что касается моделей, живёт в корне репозитория в папке models/
MODEL_DIR = Path(__file__).parent.parent / "models" / "fine_tuned_model"
# Результаты сохраняем в корне репозитория
RESULTS_PATH = Path(__file__).parent.parent / "fine_tuned_results.txt"

# Данные и модель — конвейер задач 2-4
df = load_sentiment_dataset()
texts = df["text"].tolist()
labels = [int(label) for label in df["label"]]
num_labels = len(set(labels))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = get_tokenizer(EN_MODEL)
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=MAX_LENGTH)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL, num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Обучение — как в задаче 7
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_acc, val_f1 = evaluate(model, val_loader, device)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print("-" * 50)

# 1. Сохранение модели и токенизатора
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"\nМодель сохранена: {MODEL_DIR}")

# 2. Сохранение финальных метрик
with open(RESULTS_PATH, "w") as f:
    f.write(f"Final Validation F1: {val_f1:.4f}\n")
    f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
print(f"Метрики сохранены: {RESULTS_PATH}")
