import pandas as pd

from common import (
    load_fine_tuned_model,
    load_sentiment_split,
    predict_fine_tuned,
)

# Загрузка тестовых данных
print("Загрузка тестовых данных...")
train_texts, val_texts, train_labels, val_labels = load_sentiment_split()
print(f"Тестовых примеров: {len(val_texts)}")

# Загрузка fine-tuned модели
print("\nЗагрузка fine-tuned модели...")
model_ft, tokenizer_ft = load_fine_tuned_model()

# Получение предсказаний
print("\nПолучение предсказаний...")
preds_ft_all = predict_fine_tuned(val_texts, model_ft, tokenizer_ft)
y_pred_ft = [p["prediction"] for p in preds_ft_all]

# Создание DataFrame с результатами
df_test = pd.DataFrame(
    {
        "text": val_texts,
        "true_label": val_labels,
        "pred_label": y_pred_ft,
    }
)

# Анализ ошибок
errors = df_test[df_test["true_label"] != df_test["pred_label"]]

# False Positives: предсказала positive (1), а было negative (0)
fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]

# False Negatives: предсказала negative (0), а было positive (1)
fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

print("\n" + "=" * 80)
print("=== Статистика ошибок ===")
print("=" * 80)
print(f"Всего примеров: {len(df_test)}")
print(f"Правильных предсказаний: {len(df_test) - len(errors)}")
print(f"Всего ошибок: {len(errors)}")
print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

# Примеры False Positives
if len(fp) > 0:
    print("\n" + "=" * 80)
    print("=== Примеры False Positives (предсказано POSITIVE, было NEGATIVE) ===")
    print("=" * 80)
    for idx, row in fp.head(5).iterrows():
        print(f"\nТекст: {row['text']}")
        print("Истинная метка: NEGATIVE (0)")
        print("Предсказание: POSITIVE (1)")

# Примеры False Negatives
if len(fn) > 0:
    print("\n" + "=" * 80)
    print("=== Примеры False Negatives (предсказано NEGATIVE, было POSITIVE) ===")
    print("=" * 80)
    for idx, row in fn.head(5).iterrows():
        print(f"\nТекст: {row['text']}")
        print("Истинная метка: POSITIVE (1)")
        print("Предсказание: NEGATIVE (0)")

# Общая точность
accuracy = (len(df_test) - len(errors)) / len(df_test)
print("\n" + "=" * 80)
print(f"Общая точность: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 80)
