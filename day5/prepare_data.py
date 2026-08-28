from collections import Counter

from sklearn.model_selection import train_test_split

from common import EN_MODEL, SentimentDataset, get_tokenizer, load_sentiment_dataset

# День 5, задача 2: загрузка и подготовка данных
# Датасет — локальный 'Sentiment Labelled Sentences' (UCI): вместо
# pd.read_csv('your_dataset.csv') из задания используем общий
# загрузчик load_sentiment_dataset (три TSV-файла: amazon, imdb, yelp).

# 1. Загрузка датасета
df = load_sentiment_dataset()
texts = df["text"].tolist()
labels = [int(label) for label in df["label"]]

print(f"Датасет: {len(texts)} текстов")
print(f"Баланс классов: {dict(Counter(labels))}")
print(f"Источники: {dict(df['source'].value_counts())}")

# 2. Разделение 80/20: стратификация сохраняет баланс классов,
# random_state делает разбиение воспроизводимым
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

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
