from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from common import EN_MODEL, get_embeddings, get_model, get_tokenizer, load_sentiment_split

# День 4, задача 3: Logistic Regression на CLS-эмбеддингах
# Датасет: локальный датасет sentiment labelled sentences (amazon, imdb, yelp)
# с бинарной разметкой сентимента (0 = negative, 1 = positive).

RESULTS_PATH = Path(__file__).parent / "baseline_results.txt"


# 1-2. Загрузка датасета (уже разделён на train/val)
train_texts, val_texts, train_labels, val_labels = load_sentiment_split()
print(f"Датасет: {len(train_texts) + len(val_texts)} текстов")
print(f"Train: {len(train_texts)}, val: {len(val_texts)}")

# 3. CLS-эмбеддинги для train и val
tokenizer = get_tokenizer(EN_MODEL)
model = get_model(EN_MODEL)
model.eval()

print("\nИзвлекаю эмбеддинги...")
X_train = get_embeddings(train_texts, tokenizer, model)
X_val = get_embeddings(val_texts, tokenizer, model)
print(f"Train эмбеддинги: {X_train.shape}")
print(f"Val эмбеддинги: {X_val.shape}")

# 5-7. Логистическая регрессия: обучение
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)

# 8. Предсказания на валидации
y_pred = clf.predict(X_val)

# 9-11. Отчёт и macro F1
report = classification_report(val_labels, y_pred, target_names=["negative", "positive"])
print(f"\n{report}")
f1 = f1_score(val_labels, y_pred, average="macro")
print(f"Macro F1: {f1:.4f}")

# 12. Сохранение результатов
RESULTS_PATH.write_text(
    "Baseline: Logistic Regression на CLS-эмбеддингах (локальный датасет)\n\n"
    f"Train: {len(X_train)}, val: {len(X_val)}\n\n"
    f"{report}\n"
    f"Macro F1: {f1:.4f}\n"
)
print(f"\nРезультаты сохранены: {RESULTS_PATH}")

# 13. Сохранение модели для повторного использования
MODEL_PATH = Path(__file__).parent.parent / "models" / "baseline_model" / "model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"Модель сохранена: {MODEL_PATH}")
