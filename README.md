# Transformers

Практика по трансформерам и HuggingFace Transformers. Каждый день — отдельная тема с заданиями.

## Установка

```bash
# Зависимости управляются через uv
uv sync
```

## Запуск

```bash
# День 1: Токенизация
uv run python -m day1.tokenizer
uv run python -m day1.batch
uv run python -m day1.special_tokens
uv run python -m day1.explain

# День 2: Загрузка модели и hidden states
uv run python -m day2.model_loading
uv run python -m day2.hidden_states
uv run python -m day2.embeddings

# День 3: Механизм внимания
uv run python -m day3.attention
uv run python -m day3.attention_weights
uv run python -m day3.attention_vis
uv run python -m day3.attention_layers
uv run python -m day3.attention_heads
uv run python -m day3.attention_keywords

# День 4: Эмбеддинги для классификации
uv run python -m day4.tokenization
uv run python -m day4.cls_embeddings
uv run python -m day4.logistic_regression
```

## Структура

```
common.py          # Общие утилиты: загрузка токенизатора/модели, токенизация, get_embeddings, similarity
day1/              # День 1: Архитектура трансформеров и токенизация
  tokenizer.py     #   Загрузка токенизатора, токенизация и декодирование
  batch.py         #   Токенизация батчей с padding/truncation
  special_tokens.py#   Специальные токены: [CLS], [SEP], [PAD], [UNK], [MASK]
  explain.py       #   Наглядная демонстрация subword-токенизации
day2/              # День 2: Загрузка и архитектура моделей
  model_loading.py #   Загрузка моделей, просмотр архитектуры
  hidden_states.py #   Получение hidden states, извлечение CLS-embedding
  embeddings.py    #   Функция get_embeddings, cosine similarity
day3/              # День 3: Механизм внимания и его визуализация
  attention.py     #   Загрузка модели с output_attentions, токенизация, форма attention
  attention_weights.py # Извлечение attention весов: слой -> голова -> матрица [seq_len, seq_len]
  attention_vis.py #   Heatmap-визуализация attention: выбор слоя и головы
  attention_layers.py # Сравнение слоёв: энтропия как мера сфокусированности, траектория слов
  attention_heads.py # Специализация голов слоя: перепись энтропий + сетка всех 12 голов
  attention_keywords.py # Внимание к ключевому слову: направления + сканирование всех голов
day4/              # День 4: Эмбеддинги для классификации
  tokenization.py  #   Токенизация батча: padding, truncation, attention mask
  cls_embeddings.py #  Извлечение CLS-эмбеддингов батчами, проверка размерности
  logistic_regression.py # Бейзлайн на SST2: логрегрессия на CLS-эмбеддингах
  baseline_results.txt # Отчёт и macro F1 бейзлайна (создаётся скриптом)
```

## Зависимости

- **transformers** — HuggingFace Transformers (токенизаторы, модели)
- **torch** — PyTorch (тензоры, обучение моделей)
- **scikit-learn** — cosine similarity, классификация и метрики
- **matplotlib** — графики и отображение окон
- **seaborn** — heatmap-визуализация attention
- **datasets** — загрузка датасетов с HF Hub (SST2)
- **pandas** — DataFrame для работы с датасетами
