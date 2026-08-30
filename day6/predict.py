import joblib
import torch
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from common import MODEL_BASELINE_PATH, MODEL_FT_DIR, get_embeddings
from typings import PredictionResult


def predict_fine_tuned(
    texts: str | list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> list[PredictionResult]:
    """Функция предсказания для fine-tuned модели.

    Args:
        texts: строка или список строк для предсказания
        model: fine-tuned модель (AutoModelForSequenceClassification)
        tokenizer: токенизатор для модели

    Returns:
        Список словарей с предсказаниями:
        - text: исходный текст
        - prediction: предсказанный класс (0 или 1)
        - probabilities: вероятности [P(negative), P(positive)]
    """
    if isinstance(texts, str):
        texts = [texts]

    predictions: list[PredictionResult] = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())

        predictions.append(
            {"text": text, "prediction": pred, "probabilities": probs[0].cpu().numpy()}
        )

    return predictions


def predict_baseline(
    texts: str | list[str],
    model: LogisticRegression,
    tokenizer: PreTrainedTokenizerBase,
    embedding_model: PreTrainedModel,
) -> list[PredictionResult]:
    """Функция предсказания для baseline модели (логрегрессия на CLS-эмбеддингах).

    Args:
        texts: строка или список строк для предсказания
        model: baseline модель (LogisticRegression)
        tokenizer: токенизатор для получения эмбеддингов
        embedding_model: модель для извлечения CLS-эмбеддингов

    Returns:
        Список словарей с предсказаниями:
        - text: исходный текст
        - prediction: предсказанный класс (0 или 1)
        - probabilities: вероятности [P(negative), P(positive)]
    """
    if isinstance(texts, str):
        texts = [texts]

    # Получение CLS-эмбеддингов
    embeddings = get_embeddings(texts, tokenizer, embedding_model)

    # Предсказания
    predictions_arr = model.predict(embeddings)
    probs_arr = model.predict_proba(embeddings)

    results: list[PredictionResult] = []
    for i, text in enumerate(texts):
        results.append(
            {
                "text": text,
                "prediction": int(predictions_arr[i]),
                "probabilities": probs_arr[i],
            }
        )

    return results


if __name__ == "__main__":
    # Загрузка fine-tuned модели
    print("Загрузка fine-tuned модели...")
    model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_FT_DIR)
    tokenizer_ft = AutoTokenizer.from_pretrained(MODEL_FT_DIR)
    model_ft.eval()

    # Загрузка baseline модели
    print("Загрузка baseline модели...")
    model_bl = joblib.load(MODEL_BASELINE_PATH)
    tokenizer_bl = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model_bl_emb = AutoModel.from_pretrained("distilbert-base-uncased")
    model_bl_emb.eval()

    # Тестовые примеры
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film, complete waste of time. The acting was horrible.",
        "An amazing story with great performances. Highly recommended!",
        "Boring and predictable. I almost fell asleep watching this.",
        "One of the best movies I've ever seen. Truly masterpiece!",
    ]

    # Предсказания fine-tuned модели
    print("\nПредсказания fine-tuned модели:")
    print("=" * 80)
    results_ft = predict_fine_tuned(test_texts, model_ft, tokenizer_ft)

    for result in results_ft:
        label = "POSITIVE" if result["prediction"] == 1 else "NEGATIVE"
        probs = result["probabilities"]
        print(f"\nТекст: {result['text']}")
        print(f"Предсказание: {label} (класс {result['prediction']})")
        print(f"Вероятности: negative={probs[0]:.4f}, positive={probs[1]:.4f}")

    # Предсказания baseline модели
    print("\n" + "=" * 80)
    print("\nПредсказания baseline модели:")
    print("=" * 80)
    results_bl = predict_baseline(test_texts, model_bl, tokenizer_bl, model_bl_emb)

    for result in results_bl:
        label = "POSITIVE" if result["prediction"] == 1 else "NEGATIVE"
        probs = result["probabilities"]
        print(f"\nТекст: {result['text']}")
        print(f"Предсказание: {label} (класс {result['prediction']})")
        print(f"Вероятности: negative={probs[0]:.4f}, positive={probs[1]:.4f}")

    # Статистика
    print("\n" + "=" * 80)
    print(f"\nВсего предсказаний: {len(results_ft)}")
    print(f"Fine-tuned - Позитивных: {sum(1 for r in results_ft if r['prediction'] == 1)}")
    print(f"Fine-tuned - Негативных: {sum(1 for r in results_ft if r['prediction'] == 0)}")
    print(f"Baseline - Позитивных: {sum(1 for r in results_bl if r['prediction'] == 1)}")
    print(f"Baseline - Негативных: {sum(1 for r in results_bl if r['prediction'] == 0)}")
