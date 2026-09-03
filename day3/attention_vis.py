from pathlib import Path

from common import EN_MODEL, RU_MODEL, forward_with_attention, visualize_attention

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

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
        # Формируем имя файла: model_short_layer{L}_head{H}.png
        model_short = "en" if "uncased" in model_name else "multi"
        save_path = ARTIFACTS_DIR / f"attention_{model_short}_layer{layer}_head{head}.png"
        visualize_attention(tokens, attentions, tokenizer, layer, head, save_path=str(save_path))
