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

# День 5, задача 6: функция для оценки
# Сама evaluate лежит в common.py — это инструмент финальной оценки:
# форвард без labels, логиты -> argmax -> accuracy и макро-F1

MAX_LENGTH = 64

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

# Оценка до обучения: случайная голова должна дать ~0.5
accuracy, macro_f1 = evaluate(model, val_loader, device)
print("До обучения (случайная голова):")
print(f"  accuracy: {accuracy:.3f} | macro F1: {macro_f1:.3f}")

# Одна эпоха обучения — чтобы оценивать было что
train_loss = train_epoch(model, train_loader, optimizer, device)
print(f"\nЭпоха 1: train loss {train_loss:.4f}")

# Оценка после обучения
accuracy, macro_f1 = evaluate(model, val_loader, device)
print("\nПосле обучения:")
print(f"  accuracy: {accuracy:.3f} | macro F1: {macro_f1:.3f}")
