import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from common import MAX_LENGTH, build_classifier, build_sentiment_loaders, train_epoch

# День 5, задача 5: функция обучения одной эпохи
# Обучение идёт на CPU, поэтому последовательности укорочены до 64 токенов:
# предложения датасета короткие (в масках было ~9-30 токенов из 128),
# а время батча растёт с длиной последовательности

EPOCHS = 1

# Данные и модель — конвейер задач 2-4
train_loader, val_loader, num_labels = build_sentiment_loaders(max_length=MAX_LENGTH)
model, optimizer, device = build_classifier(num_labels)


def eval_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Валидация без градиентов: лосс, точность и макро-F1."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            all_preds.extend(outputs.logits.argmax(dim=-1).tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(dataloader), float(accuracy), float(macro_f1)


# Метрики до обучения: у случайной головы ожидаем лосс ~0.69 и точность ~0.5
val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, device)
print("До обучения (случайная голова):")
print(f"  val loss: {val_loss:.4f} | accuracy: {val_acc:.3f} | macro F1: {val_f1:.3f}")

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, device)
    print(f"\nЭпоха {epoch}: train loss {train_loss:.4f}")
    print(f"  val loss: {val_loss:.4f} | accuracy: {val_acc:.3f} | macro F1: {val_f1:.3f}")
