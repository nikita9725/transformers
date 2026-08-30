import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from common import MODEL_FT_DIR
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


if __name__ == "__main__":
    # Загрузка fine-tuned модели
    print("Загрузка fine-tuned модели...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FT_DIR)
    model.eval()

    # Тестовые примеры
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film, complete waste of time. The acting was horrible.",
        "An amazing story with great performances. Highly recommended!",
        "Boring and predictable. I almost fell asleep watching this.",
        "One of the best movies I've ever seen. Truly masterpiece!",
    ]

    # Предсказания
    print("\nПредсказания fine-tuned модели:")
    print("=" * 80)
    results = predict_fine_tuned(test_texts, model, tokenizer)

    for result in results:
        label = "POSITIVE" if result["prediction"] == 1 else "NEGATIVE"
        probs = result["probabilities"]
        print(f"\nТекст: {result['text']}")
        print(f"Предсказание: {label} (класс {result['prediction']})")
        print(f"Вероятности: negative={probs[0]:.4f}, positive={probs[1]:.4f}")

    print("\n" + "=" * 80)
    print(f"Всего предсказаний: {len(results)}")
    print(f"Позитивных: {sum(1 for r in results if r['prediction'] == 1)}")
    print(f"Негативных: {sum(1 for r in results if r['prediction'] == 0)}")
