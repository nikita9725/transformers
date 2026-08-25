import torch

from common import EN_MODEL, RU_MODEL, get_model, get_tokenizer, visualize_attention

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

    model = get_model(model_name, output_attentions=True)
    model.eval()

    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)

    for layer, head in layer_heads:
        visualize_attention(tokens, outputs.attentions, tokenizer, layer, head)
