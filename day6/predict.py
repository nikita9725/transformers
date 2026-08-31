from common import (
    load_baseline_model,
    load_fine_tuned_model,
    predict_baseline,
    predict_fine_tuned,
)

if __name__ == "__main__":
    # Загрузка fine-tuned модели
    print("Загрузка fine-tuned модели...")
    model_ft, tokenizer_ft = load_fine_tuned_model()

    # Загрузка baseline модели
    print("Загрузка baseline модели...")
    model_bl, tokenizer_bl, model_bl_emb = load_baseline_model()

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
