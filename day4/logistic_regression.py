from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from common import EN_MODEL, get_embeddings, get_model, get_tokenizer, load_sst2_sample

# День 4, задача 3: Logistic Regression на CLS-эмбеддингах
# Датасет: SST2 (Stanford Sentiment Treebank) — рецензии с бинарной
# разметкой сентимента (0 = negative, 1 = positive).
# Берём сбалансированный срез по N_PER_CLASS текстов на класс:
# полного датасета (67 тысяч) для линейного бейзлайна не нужно,
# а время извлечения эмбеддингов растёт линейно.

N_PER_CLASS = 1000
RESULTS_PATH = Path(__file__).parent / "baseline_results.txt"


# 1-2. Датасет: DataFrame с текстом и меткой, списки текстов и меток
df = load_sst2_sample(N_PER_CLASS)
print(f"Датасет: {len(df)} текстов")
print(f"Баланс классов:\n{df['label'].value_counts().to_string()}")

texts = df["text"].tolist()
y = df["label"].to_numpy()

# 3. CLS-эмбеддинги всех текстов (get_embeddings из common.py —
# аналог get_cls_embeddings из задания: батчи, no_grad, CLS, vstack)
tokenizer = get_tokenizer(EN_MODEL)
model = get_model(EN_MODEL)
model.eval()

print("\nИзвлекаю эмбеддинги...")
X = get_embeddings(texts, tokenizer, model)
print(f"Эмбеддинги: {X.shape}")  # [n_texts, 768]

# 4. Разделение 80/20 со стратификацией (баланс классов в обеих частях)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)}, test: {len(X_test)}")

# 5-7. Логистическая регрессия: обучение
# В задании был n_jobs=-1, но в sklearn >= 1.8 он на логрегрессию
# не влияет и помечен как устаревший — поэтому без него
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 8. Предсказания на тесте
y_pred = clf.predict(X_test)

# 9-11. Отчёт и macro F1
report = classification_report(y_test, y_pred, target_names=["negative", "positive"])
print(f"\n{report}")
f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1: {f1:.4f}")

# 12. Сохранение результатов
RESULTS_PATH.write_text(
    "Baseline: Logistic Regression на CLS-эмбеддингах (SST2)\n\n"
    f"Train: {len(X_train)}, test: {len(X_test)}\n\n"
    f"{report}\n"
    f"Macro F1: {f1:.4f}\n"
)
print(f"\nРезультаты сохранены: {RESULTS_PATH}")
