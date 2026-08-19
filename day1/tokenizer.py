from common import EN_MODEL, RU_MODEL, get_tokenizer

examples = [
    (EN_MODEL, "This movie was absolutely amazing!"),
    (RU_MODEL, "Этот фильм был потрясающим!"),
]

for model_name, text in examples:
    print(f"\nModel: {model_name}")
    print("=" * 50)

    tokenizer = get_tokenizer(model_name)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max length: {tokenizer.model_max_length}")

    tokens = tokenizer(text)

    print(f"\nText: {text}")
    print(f"Input IDs: {tokens['input_ids']}")
    print(f"Количество токенов: {len(tokens['input_ids'])}")
    print(f"Декодировано: {tokenizer.decode(tokens['input_ids'])}")
