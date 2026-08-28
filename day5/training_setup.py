import torch
from sklearn.metrics import accuracy_score, f1_score

from common import build_classifier, build_sentiment_loaders

# День 5, задача 4: настройка обучения
# Оптимизатор берём из torch.optim: transformers.AdamW устарел —
# он давно идентичен пайторчевскому и помечен deprecated

# Данные и модель — конвейер задач 2-3
train_loader, val_loader, num_labels = build_sentiment_loaders(max_length=128)
print(f"Train: {len(train_loader)} батчей, val: {len(val_loader)} батчей")

model, optimizer, device = build_classifier(num_labels)

# 1. Метрики: ими будем мерить качество после каждой эпохи.
# Демонстрация на игрушечном примере, чтобы было видно, что они считают
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
print("\nДемонстрация метрик на игрушечном примере:")
print(f"  y_true = {y_true}, y_pred = {y_pred}")
print(f"  accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"  macro F1: {f1_score(y_true, y_pred, average='macro'):.2f}")

# 2. Оптимизатор: lr=2e-5 — типичный диапазон файн-тюнинга (1e-5..5e-5);
# большие значения разрушат предобученные веса
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nОптимизатор: AdamW, lr={optimizer.defaults['lr']:.0e}")
print(f"Обучаемых параметров: {trainable_params:,} (полный файн-тюнинг, ничего не заморожено)")

# 3. Устройство: на этом компьютере нет CUDA, поэтому model остаётся на CPU
print(f"Устройство: {device}")

# Санити-чек: первый батч на устройстве — форма логитов и лосс необученной головы.
# Если в батче есть labels, модель сама считает CrossEntropyLoss.
# ln(2) ≈ 0.693 — лосс случайного угадывания на двух классах:
# обучение должно опустить его заметно ниже этой планки
batch = next(iter(train_loader))
batch = {key: value.to(device) for key, value in batch.items()}
with torch.no_grad():
    outputs = model(**batch)
print(f"\nПервый батч на {device}:")
print(f"  форма логитов: {tuple(outputs.logits.shape)}")
print(f"  лосс: {float(outputs.loss):.4f} (случайная голова ≈ ln(2) ≈ 0.693)")
