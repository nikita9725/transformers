import torch

from common import EN_MODEL, RU_MODEL, get_model, get_tokenizer

# Задача 1: загрузка модели с attention
# Ключевой момент: output_attentions=True — без этого флага модель
# НЕ возвращает матрицы внимания (они дорогие: heads x L x L на каждый слой).

examples = [
    (EN_MODEL, "English model (distilbert-base-uncased)", "The amazing movie won many awards"),
    (
        RU_MODEL,
        "Multilingual model (distilbert-base-multilingual-cased)",
        "Потрясающий фильм получил много наград",
    ),
]

for model_name, description, text in examples:
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    # 1. Загружаем модель с output_attentions=True (важно!)
    model = get_model(model_name, output_attentions=True)
    model.eval()

    # 2. Токенизируем текст, сразу в виде PyTorch-тензоров
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer(text, return_tensors="pt")

    print(f"\nText: {text}")
    print(f"Input IDs: {tokens['input_ids']}")
    print(f"Attention mask: {tokens['attention_mask']}")

    # Расшифруем токены — это понадобится, чтобы сопоставить
    # строки/столбцы матрицы внимания конкретным словам
    ids = tokens["input_ids"][0].tolist()
    decoded = [tokenizer.decode([i]) for i in ids]
    print(f"\nTokens ({len(ids)}):")
    for i, tok in zip(ids, decoded):
        print(f"  {i:6d} -> {tok!r}")

    # Прогон без градиентов: проверяем, что attention действительно возвращается
    with torch.no_grad():
        outputs = model(**tokens)

    print(f"\nType of outputs: {type(outputs).__name__}")
    print(f"outputs.attentions is not None: {outputs.attentions is not None}")

    if outputs.attentions is not None:
        print(f"Number of layers with attention: {len(outputs.attentions)}")
        first_layer = outputs.attentions[0]
        print(f"Shape of one layer's attention: {tuple(first_layer.shape)}")
        print("  -> [batch_size, num_heads, seq_len, seq_len]")
