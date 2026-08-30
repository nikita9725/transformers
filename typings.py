"""Общие типы проекта."""

from typing import TypedDict

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

# Веса внимания всех слоёв: по тензору [batch, heads, seq_len, seq_len] на слой
Attentions = tuple[torch.Tensor, ...]


class ModelInput(TypedDict):
    """Элемент датасета: то, что возвращает SentimentDataset.__getitem__."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class PredictionResult(TypedDict):
    """Результат предсказания для одного текста."""

    text: str
    prediction: int
    probabilities: np.ndarray


# Результат сканирования голов: (вес, слой, голова, токен)
HeadLink = tuple[float, int, int, str]

# Сплит датасета: (тексты трейна, тексты вала, метки трейна, метки вала)
TextSplit = tuple[list[str], list[str], list[int], list[int]]

# Загрузчики для обучения: (трейн-лоадер, вал-лоадер, число классов)
LoadersBundle = tuple[DataLoader, DataLoader, int]

# Модель для классификации в сборе: (модель, оптимизатор, устройство)
ClassifierBundle = tuple[PreTrainedModel, Optimizer, torch.device]
