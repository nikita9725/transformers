from common import build_classifier, build_sentiment_loaders

# День 5, задача 3: загрузка модели для классификации
# AutoModelForSequenceClassification — трансформер с классификационной
# головой: для каждого текста выдаёт логиты (оценки) по одному на класс.

# 1. Число классов — по меткам датасета
train_loader, val_loader, num_labels = build_sentiment_loaders(max_length=128)
print(f"Количество классов: {num_labels}")

model, _optimizer, _device = build_classifier(num_labels)
print(f"\nКласс модели: {type(model).__name__}")

# Тело трансформера загрузилось из предобученного чекпоинта, а
# классификационная голова создана со случайными весами. Именно про это
# стандартное предупреждение transformers "Some weights were not
# initialized" (у нас оно подавлено настройками логирования) —
# голову и дообучает файн-тюнинг.
# Голова DistilBERT — это слои pre_classifier и classifier; берём их
# через getattr, потому что в типе PreTrainedModel их нет
total_params = sum(p.numel() for p in model.parameters())
head_params = sum(p.numel() for p in getattr(model, "pre_classifier").parameters()) + sum(
    p.numel() for p in getattr(model, "classifier").parameters()
)
print(f"Параметров всего: {total_params:,}")
print(f"Из них голова классификации: {head_params:,} ({head_params / total_params:.1%})")

# 2. DataLoaders: shuffle только у трейна — порядок примеров не должен
# нести информацию при обучении; валидация идёт детерминированно
# У .dataset нет __len__ в тайпингах torch, поэтому игнорируем
train_size = len(train_loader.dataset)  # type: ignore[arg-type]
val_size = len(val_loader.dataset)  # type: ignore[arg-type]
print(f"\nTrain: {len(train_loader)} батчей по 16 ({train_size} примеров)")
print(f"Val:   {len(val_loader)} батчей по 16 ({val_size} примеров)")

# Первый батч из train_loader: SentimentDataset токенизирует тексты на лету,
# а словари отдельных примеров склеиваются в батч-тензоры
batch = next(iter(train_loader))
print("\nПервый батч из train_loader:")
for key, value in batch.items():
    print(f"  {key}: {tuple(value.shape)}, dtype {value.dtype}")
print(f"  Метки батча: {batch['labels'].tolist()}")
