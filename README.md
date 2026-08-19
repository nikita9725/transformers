# Transformers

Практика по трансформерам и HuggingFace Transformers. Каждый день — отдельная тема с заданиями.

## Установка

```bash
# Зависимости управляются через uv
uv sync
```

## Запуск

```bash
# Запуск скрипта как модуля (из корня проекта)
uv run python -m day1.tokenizer
uv run python -m day1.batch
uv run python -m day1.special_tokens
uv run python -m day1.explain
```

## Структура

```
common.py          # Общие утилиты: загрузка токенизатора, токенизация батчей, explain
day1/              # День 1: Архитектура трансформеров и токенизация
  tokenizer.py     #   Загрузка токенизатора, токенизация и декодирование
  batch.py         #   Токенизация батчей с padding/truncation
  special_tokens.py#   Специальные токены: [CLS], [SEP], [PAD]
  explain.py       #   Наглядная демонстрация subword-токенизации
```

## Зависимости

- **transformers** — HuggingFace Transformers (токенизаторы, модели)
- **torch** — PyTorch (тензоры, обучение моделей)
