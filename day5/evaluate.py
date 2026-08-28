from common import build_classifier, build_sentiment_loaders, evaluate, train_epoch

# День 5, задача 6: функция для оценки
# Сама evaluate лежит в common.py — это инструмент финальной оценки:
# форвард без labels, логиты -> argmax -> accuracy и макро-F1

MAX_LENGTH = 64

# Данные и модель — конвейер задач 2-4
train_loader, val_loader, num_labels = build_sentiment_loaders(max_length=MAX_LENGTH)
model, optimizer, device = build_classifier(num_labels)

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
