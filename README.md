# Sentiment Analysis с BERT

Практика по трансформерам и HuggingFace Transformers. Каждый день — отдельная тема с заданиями.
Проект по анализу тональности текстов с использованием трансформеров (DistilBERT):
от токенизации и механизма внимания до файн-тюнга и сравнения с бейзлайном.

## Описание

Проект включает 7 учебных дней, посвящённых изучению трансформеров, и финальное
Gradio-демо для анализа тональности. На основе DistilBERT строится полный pipeline:
токенизация → эмбеддинги → механизм внимания → файн-тюнинг → сравнение моделей → анализ ошибок.

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

# День 5: Файн-тюнинг трансформера
uv run python -m day5.dataset
uv run python -m day5.prepare_data
uv run python -m day5.model_loading
uv run python -m day5.training_setup
uv run python -m day5.train_epoch
uv run python -m day5.evaluate
uv run python -m day5.train_model
uv run python -m day5.save_model

# День 6: Сравнение baseline и fine-tuned моделей
uv run python -m day6.compare_models
uv run python -m day6.predict
uv run python -m day6.confusion_matrix
uv run python -m day6.compare_metrics

# День 7: Анализ ошибок
uv run python -m day7.error_analysis

# Веб-приложение (Gradio)
uv run python app.py
```

## Структура

```
common.py          # Общие утилиты: токенизатор/модель, эмбеддинги, датасеты, хелперы обучения
typings.py         # Общие типы: ModelInput, Attentions, HeadLink, TextSplit, LoadersBundle, ClassifierBundle
app.py             # Веб-интерфейс для анализа тональности
fine_tuned_results.txt # Финальные метрики дообученной модели (создаётся day5/save_model.py)
models/            # Все сохранённые модели (baseline и fine-tuned)
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
day5/              # День 5: Файн-тюнинг трансформера
  dataset.py       #   SentimentDataset: проверка элемента датасета на примерах из датасета
  prepare_data.py  #   Загрузка локального датасета, сплит 80/20, train/val датасеты
  model_loading.py #   Модель для классификации + DataLoaders: параметры, батчи, формы
  training_setup.py #  Настройка обучения: оптимизатор, устройство, лосс случайной головы
  train_epoch.py   #   Цикл обучения одной эпохи + валидация (лосс, accuracy, macro F1)
  evaluate.py      #   Функция оценки: accuracy и macro F1 до и после обучения
  train_model.py   #   Полное обучение: 3 эпохи с оценкой после каждой
  save_model.py    #   Обучение + сохранение чекпоинта в models/; метрики — в корень
day6/              # День 6: Сравнение baseline и fine-tuned моделей
  compare_models.py #   Сравнение логрегрессии (день 4) и fine-tuned (день 5) на одном тесте
  predict.py       #   Функции предсказания для fine-tuned и baseline моделей
  confusion_matrix.py # Confusion matrix для обеих моделей с визуализацией
  compare_metrics.py #  Детальное сравнение метрик: classification_report, F1, accuracy
day7/              # День 7: Анализ ошибок
  error_analysis.py #   Анализ False Positive и False Negative с примерами ошибок
```

## Результаты

### Fine-tuned модель (DistilBERT, 3 эпохи):
- F1 (macro): 0.9115
- Accuracy: 0.9117

### Baseline модель (логистическая регрессия на CLS-эмбеддингах):
- F1 (macro): 0.8967
- Accuracy: 0.8967

Улучшение F1: **1.65%**

### Анализ ошибок
- Всего примеров: 600, ошибок: 53 (точность 91.17%)
- False Positives: 41 — модель ошибочно считает негативные отзывы позитивными
- False Negatives: 12 — модель пропускает позитивные отзывы (неявный позитив, идиомы)

## Использование в коде

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained('./models/fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./models/fine_tuned_model')

# Предсказание
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

## Зависимости

- **transformers** — HuggingFace Transformers (токенизаторы, модели)
- **torch** — PyTorch (тензоры, обучение моделей)
- **gradio** — веб-интерфейс для анализа тональности
- **scikit-learn** — cosine similarity, классификация и метрики
- **matplotlib** — графики и отображение окон
- **seaborn** — heatmap-визуализация attention
- **datasets** — загрузка датасетов с HF Hub (SST2)
- **pandas** — DataFrame для работы с датасетами
