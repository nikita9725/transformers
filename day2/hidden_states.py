import torch

from common import EN_MODEL, RU_MODEL, get_model, get_tokenizer

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

    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text, return_tensors="pt")

    # Прогоняем через модель без градиентов (экономия памяти при инференсе)
    with torch.no_grad():
        outputs = model(**tokens)

    # outputs — BaseModelOutput с last_hidden_state
    print(f"\nType: {type(outputs)}")
    print(f"Shape: {outputs.last_hidden_state.shape}")
    # [batch_size, sequence_length, hidden_size]

    # CLS-токен (первый токен) — агрегирует информацию обо всём тексте
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(f"\nCLS embedding shape: {cls_embedding.shape}")
    print(f"CLS embedding (first 5): {cls_embedding[0][:5]}")
