from common import EN_MODEL, RU_MODEL, forward_with_attention

# Задача 2: получение attention весов
# Достаём из outputs.attentions конкретную матрицу:
# слой -> голова -> матрица [seq_len, seq_len]

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

    # 1. Прогоняем текст через модель с output_attentions=True
    attentions, tokens, tokenizer = forward_with_attention(text, model_name)

    # 2. Изучаем attention
    print(f"\nType: {type(attentions)}")
    print(f"Количество слоёв: {len(attentions)}")
    print(f"Форма attention для слоя 0: {attentions[0].shape}")

    # 3. Извлекаем attention из первого слоя
    attention = attentions[0]  # первый слой
    print(f"\nAttention shape: {attention.shape}")

    # Для первого батча, первой головы
    attn_single = attention[0, 0]  # [seq_len, seq_len]
    print(f"Single head shape: {attn_single.shape}")

    # Подписываем строки и столбцы матрицы токенами:
    # позиция i в последовательности = i-я строка/столбец матрицы
    token_ids = tokens["input_ids"][0].tolist()
    token_strings = [tokenizer.decode([i]) for i in token_ids]

    print("\nAttention matrix (layer 0, head 0):")
    print(f"{'':>12}" + "".join(f"{t:>10}" for t in token_strings))
    for i, tok in enumerate(token_strings):
        row = "".join(f"{attn_single[i, j]:>10.3f}" for j in range(len(token_strings)))
        print(f"{tok:>12}{row}")

    # Сумма каждой строки = 1: внимание токена распределено
    # по всем токенам как распределение вероятностей (softmax)
    print(f"\nСуммы строк: {attn_single.sum(dim=1)}")
