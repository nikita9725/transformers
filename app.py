import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import torch

from common import load_fine_tuned_model

# Загрузка модели при старте
print("Загрузка модели...")
model, tokenizer = load_fine_tuned_model()
print("Модель загружена!")


def predict_sentiment(text: str) -> str:
    """Функция для Gradio интерфейса.

    Args:
        text: текст для анализа тональности

    Returns:
        Строка с предсказанием и вероятностями
    """
    if not text.strip():
        return "Пожалуйста, введите текст для анализа."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    # Переносим входы на устройство модели (CPU или CUDA)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = int(torch.argmax(probs, dim=1).item())

    label_map = {0: "Negative", 1: "Positive"}

    result = f"Prediction: {label_map[pred]}\n\n"
    result += "Probabilities:\n"
    result += f"Negative: {probs[0][0] * 100:.2f}%\n"
    result += f"Positive: {probs[0][1] * 100:.2f}%\n"

    return result


# Создаём интерфейс
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с DistilBERT",
    description="Введите текст и модель определит его тональность (Negative/Positive)",
)

if __name__ == "__main__":
    demo.launch()
