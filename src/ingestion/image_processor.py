# src/ingestion/image_processor.py
from pathlib import Path

from PIL import Image, ImageOps

from config import MAX_IMAGE_SIZE


def load_and_normalize(image_path: str) -> Image.Image:
    """
    Open an image, fix orientation from EXIF data, convert to RGB,
    and resize if it is bigger than MAX_IMAGE_SIZE on the long side.
    """
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)  # phones store rotation in metadata, not pixels

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = resize_if_needed(image)
    return image


def resize_if_needed(image: Image.Image, max_dim: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Downscale so the longest side is at most max_dim. Never upscales."""
    width, height = image.size
    longest = max(width, height)

    if longest <= max_dim:
        return image

    scale = max_dim / longest
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def save_processed_image(image: Image.Image, source_name: str, output_dir: str) -> str:
    """Save a normalized image, returns the saved path as a string."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = Path(source_name).stem
    file_path = output_path / f"{stem}_processed.png"
    image.save(file_path)
    return str(file_path)