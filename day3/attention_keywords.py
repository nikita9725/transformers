import torch

from common import forward_with_attention, get_token_list, visualize_attention

# Задача 6: анализ внимания к ключевым словам
# Ищем, куда направлено внимание сентиментного слова "terrible" и кто смотрит на него.
# Сканирование ведём только по содержательным токенам (без [CLS]/[SEP] и диагонали):
# из задач 4-5 знаем, что иначе всё заглушат стоки в спецтокены.

TEXT = "This movie was absolutely terrible and I hated it"
KEYWORD = "terrible"


def find_keyword_indices(token_list: list[str], keyword: str) -> list[int]:
    """Возвращает индексы токенов, образующих ключевое слово (учитывает субтокены)."""
    kw = keyword.lower()
    for start in range(len(token_list)):
        built = ""
        indices: list[int] = []
        for j in range(start, len(token_list)):
            built += token_list[j].lower().replace("##", "")
            indices.append(j)
            if built == kw:
                return indices
            if len(built) >= len(kw):
                break
    return []


def print_top_directions(
    attn: torch.Tensor,
    keyword: str,
    keyword_indices: list[int],
    token_list: list[str],
    layer: int,
    head: int,
) -> None:
    """Печатает топ-3 направления для ключевого слова: куда оно смотрит и кто смотрит на него."""
    # Строка (или сумма строк для субтокенов): куда смотрит ключевое слово
    row = attn[keyword_indices].sum(dim=0)
    print(f"\nКуда смотрит '{keyword}' (слой {layer}, голова {head}):")
    top_row = row.topk(3)
    for value, j in zip(top_row.values, top_row.indices):
        print(f"  -> {token_list[int(j)]}: {float(value):.3f}")

    # Колонка (или сумма колонок): кто смотрит на ключевое слово
    col = attn[:, keyword_indices].sum(dim=1)
    print(f"\nКто смотрит на '{keyword}' (слой {layer}, голова {head}):")
    top_col = col.topk(3)
    for value, i in zip(top_col.values, top_col.indices):
        print(f"  {token_list[int(i)]} ->: {float(value):.3f}")


def scan_heads_for_keyword(
    attn_all: torch.Tensor,
    keyword_indices: list[int],
    token_list: list[str],
) -> tuple[list[tuple[float, int, int, str]], list[tuple[float, int, int, str]]]:
    """Сканирует все головы: куда смотрит ключевое слово и кто смотрит на него.

    Учитываются только содержательные токены: [CLS], [SEP] и само слово исключены.

    Возвращает два отсортированных списка (вес, слой, голова, токен):
    первый — куда смотрит слово, второй — кто смотрит на слово.
    """
    seq_len = attn_all.shape[-1]
    forbidden = set(keyword_indices) | {0, seq_len - 1}
    key_positions = [j for j in range(seq_len) if j not in forbidden]
    query_positions = [i for i in range(seq_len) if i not in forbidden]

    sent: list[tuple[float, int, int, str]] = []
    received: list[tuple[float, int, int, str]] = []
    for layer in range(attn_all.shape[0]):
        for head in range(attn_all.shape[1]):
            matrix = attn_all[layer, head]

            row = matrix[keyword_indices].sum(dim=0)
            best_key = max(key_positions, key=lambda j: float(row[j]))
            sent.append((float(row[best_key]), layer, head, token_list[best_key]))

            col = matrix[:, keyword_indices].sum(dim=1)
            best_query = max(query_positions, key=lambda i: float(col[i]))
            received.append((float(col[best_query]), layer, head, token_list[best_query]))

    sent.sort(reverse=True)
    received.sort(reverse=True)
    return sent, received


attentions, tokens, tokenizer = forward_with_attention(TEXT)
token_list = get_token_list(tokens, tokenizer)

print(f"Text: {TEXT}")
print(f"Tokens: {token_list}")

keyword_indices = find_keyword_indices(token_list, KEYWORD)
if not keyword_indices:
    raise ValueError(f"Ключевое слово '{KEYWORD}' не найдено среди токенов: {token_list}")
found_tokens = [token_list[i] for i in keyword_indices]
print(f"Позиции '{KEYWORD}': {found_tokens} (индексы {keyword_indices})")

# Часть A: окно из задания — слой 5, голова 0
visualize_attention(tokens, attentions, tokenizer, layer=5, head=0)

# Часть B: направления внимания для слоя 5, головы 0
print_top_directions(attentions[5][0, 0], KEYWORD, keyword_indices, token_list, layer=5, head=0)

# Часть C: сканирование всех 72 голов
attn_all = torch.stack(attentions).squeeze(1)
sent, received = scan_heads_for_keyword(attn_all, keyword_indices, token_list)

print(f"\nТоп-3 головы: куда сильнее всего смотрит '{KEYWORD}' (только содержательные токены):")
for weight, layer, head, token in sent[:3]:
    print(f"  слой {layer}, голова {head:>2}: {KEYWORD} -> {token} ({weight:.3f})")

print(f"\nТоп-3 головы: кто сильнее всего смотрит на '{KEYWORD}' (только содержательные токены):")
for weight, layer, head, token in received[:3]:
    print(f"  слой {layer}, голова {head:>2}: {token} -> {KEYWORD} ({weight:.3f})")

# Окно для головы-чемпиона по направлению "куда смотрит ключевое слово"
champion_layer = sent[0][1]
champion_head = sent[0][2]
if (champion_layer, champion_head) != (5, 0):
    print(f"\nГолова-чемпион: слой {champion_layer}, голова {champion_head}")
    visualize_attention(tokens, attentions, tokenizer, layer=champion_layer, head=champion_head)
