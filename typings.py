"""Общие типы проекта."""

from typing import TypedDict

import torch

# Веса внимания всех слоёв: по тензору [batch, heads, seq_len, seq_len] на слой
Attentions = tuple[torch.Tensor, ...]


class ModelInput(TypedDict):
    """Элемент датасета: то, что возвращает SentimentDataset.__getitem__."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


# Результат сканирования голов: (вес, слой, голова, токен)
HeadLink = tuple[float, int, int, str]
