from pathlib import Path

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
        text = row["text"].strip()
        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\nТекст: {text_preview}")
        print(f"Истинный класс: {row['true_label']}, Предсказан: {row['pred_label']}")

# Примеры False Negatives
if len(fn) > 0:
    print("\n" + "=" * 80)
    print("=== Примеры False Negatives (предсказано NEGATIVE, было POSITIVE) ===")
    print("=" * 80)
    for idx, row in fn.head(5).iterrows():
        text = row["text"].strip()
        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\nТекст: {text_preview}")
        print(f"Истинный класс: {row['true_label']}, Предсказан: {row['pred_label']}")

# Анализ длины текстов
errors["text_length"] = errors["text"].str.len()
avg_error_length = errors["text_length"].mean()
avg_all_length = df_test["text"].str.len().mean()

print("\n" + "=" * 80)
print("=== Анализ длины текстов ===")
print("=" * 80)
print(f"Средняя длина ошибочных текстов: {avg_error_length:.0f} символов")
print(f"Средняя длина всех текстов: {avg_all_length:.0f} символов")
print(f"Разница: {avg_error_length - avg_all_length:+.0f} символов")

# Общая точность
accuracy = (len(df_test) - len(errors)) / len(df_test)
print("\n" + "=" * 80)
print(f"Общая точность: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 80)

# Сохранение анализа в файл
results_path = Path(__file__).parent.parent / "error_analysis.txt"
with open(results_path, "w", encoding="utf-8") as f:
    f.write("=== АНАЛИЗ ОШИБОК ===\n\n")
    f.write(f"Всего примеров: {len(df_test)}\n")
    f.write(f"Всего ошибок: {len(errors)}\n")
    f.write(f"False Positives: {len(fp)}\n")
    f.write(f"False Negatives: {len(fn)}\n")
    f.write(f"Общая точность: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")

    f.write("=== АНАЛИЗ ДЛИНЫ ТЕКСТОВ ===\n")
    f.write(f"Средняя длина ошибочных текстов: {avg_error_length:.0f} символов\n")
    f.write(f"Средняя длина всех текстов: {avg_all_length:.0f} символов\n")
    f.write(f"Разница: {avg_error_length - avg_all_length:+.0f} символов\n\n")

    f.write("=== ПРИМЕРЫ FALSE POSITIVES ===\n")
    for idx, row in fp.head(5).iterrows():
        f.write(f"\nТекст: {row['text'].strip()}\n")
        f.write(f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}\n")

    f.write("\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
    for idx, row in fn.head(5).iterrows():
        f.write(f"\nТекст: {row['text'].strip()}\n")
        f.write(f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}\n")

    f.write("\n\n=== НАБЛЮДЕНИЯ ===\n")
    f.write("Паттерны ошибок:\n")
    f.write(
        f"1. False Positives ({len(fp)} случаев): "
        "модель ошибочно считает негативные отзывы позитивными\n"
    )
    f.write(f"2. False Negatives ({len(fn)} случаев): модель пропускает позитивные отзывы\n")

    # Сравнение длин текстов для FP и FN
    if len(fp) > 0 and len(fn) > 0:
        avg_fp_length = fp["text"].str.len().mean()
        avg_fn_length = fn["text"].str.len().mean()
        f.write("\nСредняя длина текстов:\n")
        f.write(f"   - FP: {avg_fp_length:.0f} символов\n")
        f.write(f"   - FN: {avg_fn_length:.0f} символов\n")

    # Какой тип ошибок преобладает
    f.write("\nВывод:\n")
    if len(fp) < len(fn):
        f.write(
            f"   Модель лучше распознаёт негатив ({len(fp)} FP), "
            f"но пропускает больше позитива ({len(fn)} FN)\n"
        )
    elif len(fn) < len(fp):
        f.write(
            f"   Модель лучше распознаёт позитив ({len(fn)} FN), "
            f"но ошибочно помечает больше негатива как позитив ({len(fp)} FP)\n"
        )
    else:
        f.write(f"   Оба типа ошибок встречаются одинаково часто ({len(fp)} FP и {len(fn)} FN)\n")

print(f"\nАнализ сохранён: {results_path}")
