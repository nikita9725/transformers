from common import EN_MODEL, RU_MODEL, get_model

examples = [
    (EN_MODEL, "English model (distilbert-base-uncased)"),
    (RU_MODEL, "Multilingual model (distilbert-base-multilingual-cased)"),
]

for model_name, description in examples:
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    model = get_model(model_name)
    model.eval()

    print("\nModel architecture:")
    print(model)
