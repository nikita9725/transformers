from collections import Counter

from common import EN_MODEL, SentimentDataset, get_tokenizer, load_sentiment_split

# День 5, задача 2: загрузка и подготовка данных
# Датасет — локальный 'Sentiment Labelled Sentences' (UCI); сплит 80/20
# (стратифицированный, сид 42) приходит из общего хелпера

# 1-2. Датасет и разделение
train_texts, val_texts, train_labels, val_labels = load_sentiment_split()

print(f"Датасет: {len(train_texts) + len(val_texts)} текстов")
print(f"Баланс классов: {dict(Counter(train_labels + val_labels))}")
print(f"\nTrain: {len(train_texts)}, validation: {len(val_texts)}")
print(f"Баланс в train: {dict(Counter(train_labels))}")
print(f"Баланс в val:   {dict(Counter(val_labels))}")

# 3. Dataset-объекты: токенизация будет ленивой — по запросу от DataLoader
tokenizer = get_tokenizer(EN_MODEL)
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

print(f"\nlen(train_dataset) = {len(train_dataset)}")
print(f"len(val_dataset) = {len(val_dataset)}")

# Контрольный элемент из каждого датасета
for name, dataset in [("train", train_dataset), ("val", val_dataset)]:
    item = dataset[0]
    print(
        f"{name}: input_ids {tuple(item['input_ids'].shape)}, "
        f"mask {int(item['attention_mask'].sum())}/{dataset.max_length}, "
        f"label {int(item['labels'])}"
    )
