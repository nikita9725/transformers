import os

import numpy as np
import torch
from huggingface_hub.utils import disable_progress_bars
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import logging

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
