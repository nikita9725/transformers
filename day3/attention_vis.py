import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from common import EN_MODEL, RU_MODEL, get_model, get_tokenizer


def visualize_attention(
    tokens: BatchEncoding,
    attention: tuple[torch.Tensor, ...],
    tokenizer: PreTrainedTokenizerBase,
    layer: int = 0,
    head: int = 0,
) -> None:
    """
    tokens: токенизированный текст
    attention: attention weights от модели (outputs.attentions)
    tokenizer: токенизатор для подписей осей
    layer: номер слоя для визуализации
    head: номер головы для визуализации
    """
    # Получаем attention матрицу
    attn = attention[layer][0, head]  # [seq_len, seq_len]

    # Получаем токены для подписей
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    # Конспект картинки в терминал: топ-3 пары (query -> key) с максимальным весом
    print(f"\nLayer {layer}, head {head} — топ-3 пары внимания:")
    flat = attn.flatten()
    top_indices = flat.topk(3).indices
    for rank, idx in enumerate(top_indices, start=1):
        i, j = divmod(int(idx), len(token_list))
        print(f"  {rank}. {token_list[i]} -> {token_list[j]}: {attn[i, j]:.3f}")

    # Рисуем heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.show()


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
