from common import EN_MODEL, get_tokenizer, tokenize_texts

# День 4, задача 1: токенизация текстов
# Функция tokenize_texts уже есть в common.py (padding, truncation,
# max_length, return_tensors="pt"), поэтому переиспользуем её,
# а не дублируем. Токенизатор загружается через get_tokenizer
# (внутри — AutoTokenizer).

texts = [
    "This movie was fantastic!",
    "Terrible film, a complete waste of time and money.",
    "Absolutely loved it, great acting and storyline.",
]

print(f"Model: {EN_MODEL}")
print(f"Texts: {texts}")

tokenizer = get_tokenizer(EN_MODEL)  # distilbert-base-uncased

# Токенизация батча: padding до длины самого длинного текста,
# обрезка по max_length, результат — тензоры (батч для модели)
tokens = tokenize_texts(texts, tokenizer)

print(f"\nShape: {tokens['input_ids'].shape}")  # [n_texts, seq_len]
print(f"\nInput IDs:\n{tokens['input_ids']}")
print(f"\nAttention mask:\n{tokens['attention_mask']}")

# Расшифруем каждую строку, чтобы увидеть, куда встал [PAD]
print("\nТокены каждого текста:")
for i, ids in enumerate(tokens["input_ids"]):
    print(f"  [{i}] {tokenizer.convert_ids_to_tokens(ids.tolist())}")

# Маска: сколько реальных токенов и сколько паддинга в каждой строке
print("\nРеальные токены против паддинга:")
for i, mask in enumerate(tokens["attention_mask"]):
    real = int(mask.sum())
    pads = int((1 - mask).sum())
    print(f"  [{i}] реальных: {real}, паддинг: {pads}")

# Демонстрация работы параметра max_length: обрежем до 8 токенов
tokens_short = tokenize_texts(texts, tokenizer, max_length=8)
print(f"\nTruncation с max_length=8: shape {tokens_short['input_ids'].shape}")
for i, ids in enumerate(tokens_short["input_ids"]):
    print(f"  [{i}] {tokenizer.convert_ids_to_tokens(ids.tolist())}")
