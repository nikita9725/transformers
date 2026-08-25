from common import EN_MODEL, RU_MODEL, forward_with_attention, visualize_attention

examples = [
    (EN_MODEL, "English model", "The amazing movie won many awards", [(0, 0), (5, 0)]),
    (
        RU_MODEL,
        "Multilingual model",
        "Потрясающий фильм получил много наград",
        [(0, 0)],
    ),
]

for model_name, description, text, layer_heads in examples:
    print(f"\n{'=' * 60}")
    print(f"{description}: {model_name}")
    print(f"{'=' * 60}")
    print(f"Text: {text}")

    attentions, tokens, tokenizer = forward_with_attention(text, model_name)

    for layer, head in layer_heads:
        visualize_attention(tokens, attentions, tokenizer, layer, head)
