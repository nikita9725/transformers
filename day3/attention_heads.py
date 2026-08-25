import math
from typing import cast

import torch

from common import EN_MODEL, get_model, get_tokenizer, visualize_attention_heads

# Задача 5: внимание разных голов одного слоя

TEXT = "The amazing movie won many awards"


def print_head_summary(attn_layer: torch.Tensor, token_list: list[str]) -> None:
    """Печатает по каждой голове среднюю энтропию строк и самую сильную пару внимания."""
    num_heads = attn_layer.shape[0]
    seq_len = attn_layer.shape[-1]
    max_entropy = math.log(seq_len)

    # [num_heads, seq_len] — энтропия каждой строки каждой головы
    entropy = -(attn_layer * attn_layer.clamp_min(1e-9).log()).sum(dim=-1)

    print(f"Энтропия: 0 = всё внимание в одну ячейку, {max_entropy:.2f} = равномерное")
    for head in range(num_heads):
        top_idx = int(attn_layer[head].flatten().argmax())
        i, j = divmod(top_idx, len(token_list))
        avg_entropy = float(entropy[head].mean())
        print(
            f"голова {head:>2}: энтропия {avg_entropy:>5.2f} | "
            f"топ: {token_list[i]} -> {token_list[j]} ({float(attn_layer[head, i, j]):.3f})"
        )


model = get_model(EN_MODEL, output_attentions=True)
model.eval()

tokenizer = get_tokenizer(EN_MODEL)
tokens = tokenizer(TEXT, return_tensors="pt")
# convert_ids_to_tokens типизирован как Union[str, List[str]]; для списка
# идентификаторов он всегда возвращает список, поэтому сужаем тип через cast
token_list = cast(list[str], tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist()))

with torch.no_grad():
    outputs = model(**tokens)

# Слой 0 — по заданию; слой 1 — там задача 4 нашла самые острые специализации
for layer in (0, 1):
    print(f"\n{'=' * 60}")
    print(f"Слой {layer}: специализация 12 голов")
    print(f"{'=' * 60}")
    print_head_summary(outputs.attentions[layer][0], token_list)
    visualize_attention_heads(tokens, outputs.attentions, tokenizer, layer=layer)
