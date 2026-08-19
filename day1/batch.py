from common import EN_MODEL, RU_MODEL, get_tokenizer, tokenize_texts

# EN модель — английские тексты
print(f"Model: {EN_MODEL}")
print("=" * 50)

en_tokenizer = get_tokenizer(EN_MODEL)
en_texts = [
    "This movie was great!",
    "Terrible movie, waste of time.",
]

tokens = tokenize_texts(en_texts, en_tokenizer)
print(f"Texts: {en_texts}")
print(f"Shape: {tokens['input_ids'].shape}")
print(f"Attention mask:\n{tokens['attention_mask']}")
print(f"Input IDs:\n{tokens['input_ids']}")

# Multilingual модель — тексты на разных языках
print(f"\nModel: {RU_MODEL}")
print("=" * 50)

ru_tokenizer = get_tokenizer(RU_MODEL)
mixed_texts = [
    "Фильм был отличный!",
    "This movie was great!",
    "Ужасный фильм, потеря времени.",
]

tokens = tokenize_texts(mixed_texts, ru_tokenizer)
print(f"Texts: {mixed_texts}")
print(f"Shape: {tokens['input_ids'].shape}")
print(f"Attention mask:\n{tokens['attention_mask']}")
print(f"Input IDs:\n{tokens['input_ids']}")
