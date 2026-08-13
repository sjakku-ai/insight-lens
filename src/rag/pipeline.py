import uuid
from pathlib import Path

from PIL import Image
from config import UPLOAD_DIR
from src.ingestion.pdf_loader import pdf_to_images
from src.ingestion.image_processor import load_and_normalize_image

