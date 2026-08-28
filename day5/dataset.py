import pandas as pd

from common import EN_MODEL, SentimentDataset, get_tokenizer, load_sentiment_dataset

# День 5, задача 1: подготовка Dataset класса
# Сам класс SentimentDataset лежит в common.py — он понадобится
# в следующих задачах дня (DataLoader, обучение, оценка).

# Четыре предложения из локального датасета: два отрицательных, два положительных
df = load_sentiment_dataset()
negative = df[df["label"] == 0].head(2)
positive = df[df["label"] == 1].head(2)
sample = pd.concat([negative, positive], ignore_index=True)
texts = sample["text"].tolist()
labels = [int(label) for label in sample["label"]]

print(f"Датасет: {len(texts)} текста")
for text, label in zip(texts, labels):
    print(f"  [{label}] {text}")

tokenizer = get_tokenizer(EN_MODEL)
dataset = SentimentDataset(texts, labels, tokenizer, max_length=128)
print(f"\nlen(dataset) = {len(dataset)}")

# Заглядываем в первый элемент — то, что увидит DataLoader
item = dataset[0]
print(f"\nКлючи элемента: {sorted(item.keys())}")
print(f"input_ids:      shape {tuple(item['input_ids'].shape)}, dtype {item['input_ids'].dtype}")
mask = item["attention_mask"]
print(f"attention_mask: shape {tuple(mask.shape)}, dtype {mask.dtype}")
print(f"labels:         {int(item['labels'])} (dtype {item['labels'].dtype})")

# padding="max_length" добивает каждый пример до 128 токенов:
# рецензии короткие, поэтому большая часть маски — нули
real_tokens = int(mask.sum())
print(f"\nРеальных токенов: {real_tokens} из {dataset.max_length} — остальное паддинг")

token_list = tokenizer.convert_ids_to_tokens(item["input_ids"].tolist())
print(f"Начало: {token_list[:6]} ...")
print(f"Хвост:  ... {token_list[-3:]}")
