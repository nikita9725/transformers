from common import EN_MODEL, get_embeddings, get_model, get_tokenizer

# День 4, задача 2: извлечение CLS-эмбеддингов
# Функция get_embeddings из common.py уже реализует все пункты задания:
# разбиение на батчи, токенизацию, torch.no_grad(),
# outputs.last_hidden_state[:, 0, :] (CLS-токен), .cpu().numpy()
# и склейку батчей через np.vstack. Поэтому переиспользуем её,
# а не дублируем.

texts = [
    "This movie was fantastic!",
    "Terrible film, a complete waste of time and money.",
    "Absolutely loved it, great acting and storyline.",
    "Boring and predictable, I almost fell asleep.",
    "A true masterpiece with brilliant performances.",
]

print(f"Model: {EN_MODEL}")
print(f"Texts: {len(texts)}")

tokenizer = get_tokenizer(EN_MODEL)
model = get_model(EN_MODEL)
model.eval()  # режим инференса: отключает dropout и т.п.

# batch_size=2: 5 текстов -> 3 батча (2+2+1), цикл батчей реально работает
embeddings = get_embeddings(texts, tokenizer, model, batch_size=2)

print(f"\nEmbeddings shape: {embeddings.shape}")  # [n_texts, hidden_size]
print(f"Type: {type(embeddings).__name__}, dtype: {embeddings.dtype}")

print("\nПервые 5 значений каждого эмбеддинга:")
for i, text in enumerate(texts):
    print(f"  [{i}] {text[:45]:45s} -> {embeddings[i][:5]}")

# Пункт 5 задания: проверка размерности вывода
n_texts, hidden_size = embeddings.shape
print("\nПроверка размерности:")
print(f"  строк: {n_texts} (текстов: {len(texts)})")
print(f"  колонок: {hidden_size} (hidden_size модели: {model.config.hidden_size})")
assert n_texts == len(texts), "на каждый текст должен быть свой вектор"
assert hidden_size == model.config.hidden_size, "длина вектора = скрытый размер модели"
print("  проверка пройдена")
