import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from huggingface_hub.utils import disable_progress_bars
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import logging

from typings import Attentions, ModelInput

# Настраиваем окружение до импорта transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Отключаем варнинги и progress bars
logging.set_verbosity_error()
disable_progress_bars()

EN_MODEL = "distilbert-base-uncased"
RU_MODEL = "distilbert-base-multilingual-cased"


def get_tokenizer(model_name: str = EN_MODEL) -> PreTrainedTokenizerBase:
    """Загружает и кэширует токенизатор."""
    return AutoTokenizer.from_pretrained(model_name)


def get_model(model_name: str = EN_MODEL, **kwargs) -> PreTrainedModel:
    """Загружает и кэширует модель.

    Дополнительные kwargs (например, output_attentions=True) пробрасываются
    в AutoModel.from_pretrained.
    """
    return AutoModel.from_pretrained(model_name, **kwargs)


def forward_with_attention(
    text: str, model_name: str = EN_MODEL
) -> tuple[Attentions, BatchEncoding, PreTrainedTokenizerBase]:
    """Прогоняет текст через модель с output_attentions=True.

    Возвращает кортеж (attention всех слоёв, токенизированный текст, токенизатор).
    """
    model = get_model(model_name, output_attentions=True)
    model.eval()

    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)

    # Флаг output_attentions=True гарантирует, что матрицы внимания вернулись
    assert outputs.attentions is not None
    return outputs.attentions, tokens, tokenizer


def tokenize_texts(
    texts: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int = 128
) -> BatchEncoding:
    """Токенизирует список текстов с padding и truncation."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text: str, tokenizer: PreTrainedTokenizerBase) -> None:
    """Показывает как текст разбивается на subword-токены."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")


def get_token_list(tokens: BatchEncoding, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Возвращает список токенов из токенизированного текста (для подписей осей)."""
    # convert_ids_to_tokens типизирован как Union[str, List[str]]; для списка
    # идентификаторов он всегда возвращает список, поэтому сужаем тип через cast
    return cast(list[str], tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist()))


def load_sst2_sample(n_per_class: int = 1000) -> pd.DataFrame:
    """Загружает сбалансированный срез SST2 как DataFrame с колонками
    text и label (0 = negative, 1 = positive)."""
    ds: HFDataset = load_dataset("stanfordnlp/sst2", split="train")
    positive = ds.filter(lambda row: row["label"] == 1).shuffle(seed=42).select(range(n_per_class))
    negative = ds.filter(lambda row: row["label"] == 0).shuffle(seed=42).select(range(n_per_class))
    df = concatenate_datasets([positive, negative]).shuffle(seed=42).to_pandas()
    return df.rename(columns={"sentence": "text"})[["text", "label"]]


class SentimentDataset(Dataset):
    """Датасет для классификации текстов: по одному примеру возвращает
    input_ids, attention_mask и labels, готовые для модели."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> ModelInput:
        text = self.texts[idx]
        label = self.labels[idx]

        # padding="max_length": каждый пример возвращается одинаковой длины,
        # иначе DataLoader не сможет собрать примеры в батч
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # CrossEntropyLoss ожидает индексы классов в long
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_embeddings(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Получает CLS-эмбеддинги для списка текстов.

    Args:
        texts: список текстов для векторизации
        tokenizer: загруженный токенизатор
        model: загруженная модель
        batch_size: размер батча для обработки
        max_length: максимальная длина последовательности

    Returns:
        numpy array формы [n_texts, hidden_size] с CLS-эмбеддингами
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Токенизация батча
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Прогон через модель без градиентов
        with torch.no_grad():
            outputs = model(**tokens)

        # Извлечение CLS-токенов (первый токен каждого текста)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # Объединяем все батчи в один array
    return np.vstack(all_embeddings)


def similarity(
    text1: str,
    text2: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> float:
    """Вычисляет косинусное сходство между двумя текстами.

    Args:
        text1: первый текст
        text2: второй текст
        tokenizer: загруженный токенизатор
        model: загруженная модель

    Returns:
        Косинусное сходство от -1 до 1 (1 = идентичны)
    """
    # Получаем эмбеддинги для обоих текстов
    emb = get_embeddings([text1, text2], tokenizer, model)

    # Вычисляем косинусное сходство
    # emb[0:1] — первый текст (shape: [1, 768])
    # emb[1:2] — второй текст (shape: [1, 768])
    # cosine_similarity возвращает матрицу [1, 1], берём [0][0]
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]

    return float(sim)


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Энтропия каждой строки матрицы внимания — мера её сфокусированности.

    0 — всё внимание сосредоточено в одной ячейке,
    ln(seq_len) — равномерное распределение по всем токенам.
    Работает и для набора матриц: энтропия считается по последнему измерению.
    """
    return -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1)


def visualize_attention(
    tokens: BatchEncoding,
    attention: Attentions,
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
    token_list = get_token_list(tokens, tokenizer)

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


def visualize_attention_heads(
    tokens: BatchEncoding,
    attention: Attentions,
    tokenizer: PreTrainedTokenizerBase,
    layer: int = 0,
) -> None:
    """
    tokens: токенизированный текст
    attention: attention weights от модели (outputs.attentions)
    tokenizer: токенизатор для подписей осей
    layer: номер слоя, все головы которого рисуются на одной сетке
    """
    # Матрицы всех голов выбранного слоя: [num_heads, seq_len, seq_len]
    attn_layer = attention[layer][0]
    num_heads = attn_layer.shape[0]

    # Получаем токены для подписей
    token_list = get_token_list(tokens, tokenizer)

    # layout="constrained" корректно размещает общий colorbar и заголовок;
    # обычный tight_layout с ними конфликтует и выдаёт UserWarning
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), layout="constrained")
    heatmap_ax = None
    for head, ax in enumerate(axes.flat):
        if head >= num_heads:
            ax.axis("off")
            continue
        # Общая шкала цвета [0, 1] для всех голов:
        # яркость сопоставима между головами напрямую
        heatmap_ax = sns.heatmap(
            attn_layer[head].cpu().numpy(),
            xticklabels=token_list,
            yticklabels=token_list,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Head {head}")
        ax.tick_params(labelsize=7)

    # Один общий colorbar на всю сетку;
    # при хотя бы одной голове цикл всегда рисует хотя бы один хитмап
    assert heatmap_ax is not None
    # sns.heatmap возвращает Axes, а для colorbar нужен сам нарисованный
    # объект с данными (меш или изображение)
    mappable = heatmap_ax.collections[0] if heatmap_ax.collections else heatmap_ax.images[0]
    fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.6)
    fig.suptitle(f"Attention - Layer {layer}, all heads")
    plt.show()
