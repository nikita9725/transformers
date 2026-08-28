from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from common import EN_MODEL, SentimentDataset, get_tokenizer, load_sentiment_dataset

# День 5, задача 3: загрузка модели для классификации
# AutoModelForSequenceClassification — трансформер с классификационной
# головой: для каждого текста выдаёт логиты (оценки) по одному на класс.

# 1. Число классов — по меткам датасета
df = load_sentiment_dataset()
texts = df["text"].tolist()
labels = [int(label) for label in df["label"]]

num_labels = len(set(labels))  # 2 для бинарной классификации
print(f"Количество классов: {num_labels}")

model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL, num_labels=num_labels)
print(f"\nКласс модели: {type(model).__name__}")

# Тело трансформера загрузилось из предобученного чекпоинта, а
# классификационная голова создана со случайными весами. Именно про это
# стандартное предупреждение transformers "Some weights were not
# initialized" (у нас оно подавлено настройками логирования) —
# голову и дообучает файн-тюнинг.
total_params = sum(p.numel() for p in model.parameters())
head_params = sum(p.numel() for p in model.pre_classifier.parameters()) + sum(
    p.numel() for p in model.classifier.parameters()
)
print(f"Параметров всего: {total_params:,}")
print(f"Из них голова классификации: {head_params:,} ({head_params / total_params:.1%})")

# Подготовка данных (задача 2): сплит и SentimentDataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = get_tokenizer(EN_MODEL)
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

# 2. DataLoaders: shuffle только у трейна — порядок примеров не должен
# нести информацию при обучении; валидация идёт детерминированно
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print(f"\nTrain: {len(train_loader)} батчей по 16 ({len(train_dataset)} примеров)")
print(f"Val:   {len(val_loader)} батчей по 16 ({len(val_dataset)} примеров)")

# Первый батч из train_loader: SentimentDataset токенизирует тексты на лету,
# а словари отдельных примеров склеиваются в батч-тензоры
batch = next(iter(train_loader))
print("\nПервый батч из train_loader:")
for key, value in batch.items():
    print(f"  {key}: {tuple(value.shape)}, dtype {value.dtype}")
print(f"  Метки батча: {batch['labels'].tolist()}")
