import math
from typing import cast

import torch

from common import EN_MODEL, get_model, get_tokenizer, visualize_attention

# Задача 4: анализ attention для разных слоёв
# Гипотеза: на последних слоях внимание более сфокусировано.
# Проверяем её численно через энтропию строк матрицы внимания.

TEXT = "The amazing movie won many awards"


def print_focus_table(attn_all: torch.Tensor) -> None:
    """Печатает метрики сфокусированности внимания по слоям.

    Энтропия строк матрицы внимания:
    - равномерное внимание по всем токенам даёт максимум ln(seq_len);
    - всё внимание в одной ячейке даёт 0.
    Чем ниже энтропия, тем сфокусированнее внимание.
    """
    num_layers = attn_all.shape[0]
    seq_len = attn_all.shape[-1]
    max_entropy = math.log(seq_len)

    # [num_layers, num_heads, seq_len] — энтропия каждой строки каждой головы
    entropy = -(attn_all * attn_all.clamp_min(1e-9).log()).sum(dim=-1)

    print("\nФокус внимания по слоям (среднее по головам и строкам):")
    print(f"{'слой':>4} | {'энтропия':>8} | {'% от макс':>8} | {'макс. вес строки':>16}")
    for layer in range(num_layers):
        avg_entropy = float(entropy[layer].mean())
        avg_max_weight = float(attn_all[layer].max(dim=-1).values.mean())
        share = avg_entropy / max_entropy
        print(f"{layer:>4} | {avg_entropy:>8.3f} | {share:>7.1%} | {avg_max_weight:>16.3f}")

    # Компактная таблица энтропии по слоям и головам —
    # видно, что фокус неоднороден и внутри слоя
    print("\nЭнтропия по слоям и головам (среднее по строкам):")
    print(f"{'':>6}" + "".join(f"{head:>7}" for head in range(attn_all.shape[1])))
    for layer in range(num_layers):
        head_entropies = (float(entropy[layer, head].mean()) for head in range(attn_all.shape[1]))
        row = "".join(f"{value:>7.2f}" for value in head_entropies)
        print(f"слой {layer}{row}")


def print_token_journey(
    token: str, token_list: list[str], attn_all: torch.Tensor, head: int = 0
) -> None:
    """Показывает, на какой токен сильнее всего смотрит выбранный токен в каждом слое."""
    idx = token_list.index(token)
    print(f"\nКуда смотрит '{token}' (голова {head}):")
    for layer in range(attn_all.shape[0]):
        row = attn_all[layer, head, idx]
        j = int(row.argmax())
        print(f"  слой {layer}: -> {token_list[j]} ({float(row[j]):.3f})")


model = get_model(EN_MODEL, output_attentions=True)
model.eval()

tokenizer = get_tokenizer(EN_MODEL)
tokens = tokenizer(TEXT, return_tensors="pt")
# convert_ids_to_tokens типизирован как Union[str, List[str]]; для списка
# идентификаторов он всегда возвращает список, поэтому сужаем тип через cast
token_list = cast(list[str], tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist()))

with torch.no_grad():
    outputs = model(**tokens)

# Все матрицы внимания в одном тензоре: [слои, головы, seq_len, seq_len]
# squeeze(1) убирает батч-измерение — у нас один текст в батче
attn_all = torch.stack(outputs.attentions).squeeze(1)

print(f"Text: {TEXT}")

# Часть A: количественное сравнение сфокусированности по слоям
print_focus_table(attn_all)

# Часть B: траектория внимания отдельных слов по глубине модели
print_token_journey("awards", token_list, attn_all)
print_token_journey("movie", token_list, attn_all)

# Часть C: три окна из задания — первый слой, средний, последний
print("\nОткрываю окна: слой 0, слой 3, слой 5 (голова 0)")
for layer in (0, 3, 5):
    visualize_attention(tokens, outputs.attentions, tokenizer, layer=layer, head=0)
