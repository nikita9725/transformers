from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

EN_MODEL = "distilbert-base-uncased"
RU_MODEL = "distilbert-base-multilingual-cased"


def get_tokenizer(model_name: str = EN_MODEL) -> PreTrainedTokenizerBase:
    """Загружает и кэширует токенизатор."""
    return AutoTokenizer.from_pretrained(model_name)


def get_model(model_name: str = EN_MODEL) -> PreTrainedModel:
    """Загружает и кэширует модель."""
    return AutoModel.from_pretrained(model_name)


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
