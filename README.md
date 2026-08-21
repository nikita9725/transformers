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
```

## Структура

```
common.py          # Общие утилиты: загрузка токенизатора и модели, токенизация батчей, explain
day1/              # День 1: Архитектура трансформеров и токенизация
  tokenizer.py     #   Загрузка токенизатора, токенизация и декодирование
  batch.py         #   Токенизация батчей с padding/truncation
  special_tokens.py#   Специальные токены: [CLS], [SEP], [PAD], [UNK], [MASK]
  explain.py       #   Наглядная демонстрация subword-токенизации
day2/              # День 2: Загрузка и архитектура моделей
  model_loading.py #   Загрузка моделей, просмотр архитектуры
  hidden_states.py #   Получение hidden states, извлечение CLS-embedding
```

## Зависимости

- **transformers** — HuggingFace Transformers (токенизаторы, модели)
- **torch** — PyTorch (тензоры, обучение моделей)
