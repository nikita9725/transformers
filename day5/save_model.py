from pathlib import Path

from common import (
    EN_MODEL,
    MAX_LENGTH,
    SEED,
    build_classifier,
    build_sentiment_loaders,
    get_tokenizer,
    set_seed,
    train_loop,
)

# День 5, задача 8: сохранение модели
# Обучаем модель (те же 3 эпохи, что в задаче 7) и сохраняем через
# save_pretrained вместе с токенизатором: чекпоинт становится
# самодостаточным, в config.json сохраняется архитектура головы (num_labels)

NUM_EPOCHS = 3
# Всё, что касается моделей, живёт в корне репозитория в папке models/
MODEL_DIR = Path(__file__).parent.parent / "models" / "fine_tuned_model"
# Результаты сохраняем в корне репозитория
RESULTS_PATH = Path(__file__).parent.parent / "fine_tuned_results.txt"

# Фиксируем seed для воспроизводимости
set_seed(SEED)

# Данные и модель — конвейер задач 2-4
train_loader, val_loader, num_labels = build_sentiment_loaders(max_length=MAX_LENGTH)
model, optimizer, device = build_classifier(num_labels)

# Обучение — как в задаче 7
val_acc, val_f1 = train_loop(model, train_loader, val_loader, optimizer, device, NUM_EPOCHS)

# 1. Сохранение модели и токенизатора
model.save_pretrained(MODEL_DIR)
get_tokenizer(EN_MODEL).save_pretrained(MODEL_DIR)
print(f"\nМодель сохранена: {MODEL_DIR}")

# 2. Сохранение финальных метрик
with open(RESULTS_PATH, "w") as f:
    f.write(f"Final Validation F1: {val_f1:.4f}\n")
    f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
print(f"Метрики сохранены: {RESULTS_PATH}")
