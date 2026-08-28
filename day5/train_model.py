from common import build_classifier, build_sentiment_loaders, train_loop

# День 5, задача 7: обучение модели
# Полный цикл из NUM_EPOCHS эпох: после каждой — оценка на валидации.
# 3 эпохи — минимум из диапазона задания; на 1 эпохе уже ~0.93,
# дальше ожидаем плато без переобучения

NUM_EPOCHS = 3
MAX_LENGTH = 64

# Данные и модель — конвейер задач 2-4
train_loader, val_loader, num_labels = build_sentiment_loaders(max_length=MAX_LENGTH)
model, optimizer, device = build_classifier(num_labels)

train_loop(model, train_loader, val_loader, optimizer, device, NUM_EPOCHS)
