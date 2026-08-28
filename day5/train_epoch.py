import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from common import EN_MODEL, SentimentDataset, get_tokenizer, load_sentiment_dataset

# День 5, задача 5: функция обучения одной эпохи
# Обучение идёт на CPU, поэтому последовательности укорочены до 64 токенов:
# предложения датасета короткие (в масках было ~9-30 токенов из 128),
# а время батча растёт с длиной последовательности

EPOCHS = 1
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


def train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Одна эпоха обучения; возвращает средний лосс по батчам."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


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
