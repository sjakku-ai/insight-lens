# src/models/ollama_client.py
import base64
import io

import ollama
from PIL import Image

from config import OLLAMA_HOST, OLLAMA_MODEL

_client = ollama.Client(host=OLLAMA_HOST)


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8") # A = 65


def ask_about_image(image: Image.Image, question: str, model: str = None) -> str:
    """Send one page image and one question to the local VLM. Returns plain text."""
    model = model or OLLAMA_MODEL
    encoded = _image_to_base64(image)

    response = _client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [encoded],
            }
        ],
    )
    return response["message"]["content"]


def ask_text_only(question: str, model: str = None) -> str:
    """For reasoning steps that do not need an image, e.g. summarizing retrieved captions."""
    model = model or OLLAMA_MODEL
    response = _client.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response["message"]["content"]


def is_ollama_running() -> bool:
    """Quick health check before you build anything on top of this."""
    try:
        _client.list()
        return True
    except Exception:
        return False