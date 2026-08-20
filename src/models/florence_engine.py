# src/models/florence_engine.py
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from config import FLORENCE_MODEL_ID, FLORENCE_DEVICE, FLORENCE_ATTN_IMPLEMENTATION

_model = None
_processor = None


def _load():
    """Load Florence-2 once. Never call this per-request, it is slow."""
    global _model, _processor
    if _model is not None:
        return

    _model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID,
        trust_remote_code=True,
        attn_implementation=FLORENCE_ATTN_IMPLEMENTATION,
        torch_dtype=torch.float32 if FLORENCE_DEVICE == "cpu" else torch.float16,
    ).to(FLORENCE_DEVICE)

    _processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_ID, trust_remote_code=True
    )


def _run_task(image: Image.Image, task_prompt: str, text_input: str = None) -> dict:
    _load()
    prompt = task_prompt if text_input is None else task_prompt + text_input

    inputs = _processor(text=prompt, images=image, return_tensors="pt").to(FLORENCE_DEVICE)

    generated_ids = _model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = _processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed


def caption_image(image: Image.Image) -> str:
    """One sentence describing the page. Good for a quick index entry."""
    result = _run_task(image, "<MORE_DETAILED_CAPTION>")
    return result.get("<MORE_DETAILED_CAPTION>", "")


def ocr_page(image: Image.Image) -> str:
    """Plain OCR text for the whole page, no region breakdown."""
    result = _run_task(image, "<OCR>")
    return result.get("<OCR>", "")


def ocr_with_regions(image: Image.Image) -> dict:
    """OCR text plus bounding boxes, for when you need to point at a specific spot."""
    result = _run_task(image, "<OCR_WITH_REGION>")
    return result.get("<OCR_WITH_REGION>", {})


def detect_regions(image: Image.Image) -> dict:
    """Object detection, useful for charts and diagrams with distinct visual elements."""
    result = _run_task(image, "<OD>")
    return result.get("<OD>", {})