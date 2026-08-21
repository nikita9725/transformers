from common import EN_MODEL, RU_MODEL, get_embeddings, get_model, get_tokenizer, similarity

texts = [
    "This movie was absolutely amazing!",
    "Terrible film, waste of time.",
    "I loved every minute of it.",
    "Boring and predictable plot.",
    "A masterpiece of modern cinema!",
]

examples = [
    (EN_MODEL, "English model"),
    (RU_MODEL, "Multilingual model"),
]

for model_name, description in examples:
    print(f"\n{'=' * 60}")
    print(f"{description}: {model_name}")
    print(f"{'=' * 60}")

    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    model.eval()

    # Получаем эмбеддинги для всех текстов
    embeddings = get_embeddings(texts, tokenizer, model, batch_size=2)

    print(f"\nTexts: {len(texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    # [n_texts, hidden_size]

    print("\nFirst 5 values of each embedding:")
    for i, text in enumerate(texts):
        print(f"  [{i}] {text[:40]:40s} → {embeddings[i][:5]}")

    # Cosine similarity между первым и остальными текстами
    print("\nCosine similarity with text[0]:")
    for i in range(1, len(texts)):
        sim = similarity(texts[0], texts[i], tokenizer, model)
        print(f"  text[0] vs text[{i}]: {sim:.4f}")
