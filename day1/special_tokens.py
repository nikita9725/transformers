from common import EN_MODEL, RU_MODEL, get_tokenizer

for model_name in [EN_MODEL, RU_MODEL]:
    print(f"\nModel: {model_name}")
    print("=" * 50)

    tokenizer = get_tokenizer(model_name)

    # Все специальные токены BERT
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"MASK token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

    # Пример: [CLS] текст [SEP]
    text = "This movie was amazing!"
    tokens = tokenizer(text, return_tensors="pt")
    print(f"\nText: {text}")
    print(f"Input IDs: {tokens['input_ids']}")
    print(f"Decoded: {tokenizer.decode(tokens['input_ids'][0])}")

    # Пример с [MASK] — замаскированное слово
    masked_text = "This movie was [MASK]!"
    masked_tokens = tokenizer(masked_text, return_tensors="pt")
    print(f"\nMasked text: {masked_text}")
    print(f"Input IDs: {masked_tokens['input_ids']}")
    mask_position = (masked_tokens["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[
        1
    ]
    print(f"MASK position: {mask_position.item()}")

    # Пример с [UNK] — символ которого нет в словаре
    unk_text = "Hello 🎉 world"
    unk_tokens = tokenizer(unk_text)
    print(f"\nText with emoji: {unk_text}")
    print(f"Input IDs: {unk_tokens['input_ids']}")
    print(f"Decoded: {tokenizer.decode(unk_tokens['input_ids'])}")
    if tokenizer.unk_token_id in unk_tokens["input_ids"]:
        print("  → Contains [UNK] token!")
