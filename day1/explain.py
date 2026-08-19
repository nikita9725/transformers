from common import EN_MODEL, RU_MODEL, explain_tokenization, get_tokenizer

en_tokenizer = get_tokenizer(EN_MODEL)
ru_tokenizer = get_tokenizer(RU_MODEL)

print("English tokenizer:")
explain_tokenization("Transformers are amazing!", en_tokenizer)

print("\n---\n")

print("Multilingual tokenizer (English):")
explain_tokenization("Transformers are amazing!", ru_tokenizer)

print("\n---\n")

print("Multilingual tokenizer (Russian):")
explain_tokenization("Трансформеры — это мощно!", ru_tokenizer)

print("\n---\n")

print("English: 'unbelievable'")
explain_tokenization("unbelievable", en_tokenizer)

print("\nMultilingual: 'невероятно'")
explain_tokenization("невероятно", ru_tokenizer)
