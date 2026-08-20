# src/rag/pipeline.py
import uuid
from pathlib import Path

from PIL import Image

from config import UPLOADS_DIR
from src.ingestion.pdf_loader import pdf_to_images, save_page_images
from src.ingestion.image_processor import load_and_normalize, save_processed_image
from src.models.florence_engine import caption_image, ocr_page
from src.models.ollama_client import ask_about_image
from src.vectorstore.chroma_store import add_documents_batch, query


def ingest_pdf(pdf_path: str) -> int:
    """Full ingest flow for a PDF. Returns the number of pages indexed."""
    pages = pdf_to_images(pdf_path)
    pages = save_page_images(pages, str(UPLOADS_DIR / "rendered"))

    doc_ids, texts, metadatas = [], [], []

    for page in pages:
        caption = caption_image(page["image"])
        ocr_text = ocr_page(page["image"])
        combined_text = f"{caption}\n{ocr_text}"

        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        texts.append(combined_text)
        metadatas.append({
            "image_path": page["image_path"],
            "page_number": page["page_number"],
            "source_file": page["source_file"],
        })

    add_documents_batch(doc_ids, texts, metadatas)
    return len(pages)


def ingest_image(image_path: str) -> int:
    """Full ingest flow for a standalone image. Returns 1 on success."""
    image = load_and_normalize(image_path)
    saved_path = save_processed_image(image, Path(image_path).name, str(UPLOADS_DIR / "rendered"))

    caption = caption_image(image)
    ocr_text = ocr_page(image)
    combined_text = f"{caption}\n{ocr_text}"

    add_documents_batch(
        doc_ids=[str(uuid.uuid4())],
        texts=[combined_text],
        metadatas=[{
            "image_path": saved_path,
            "page_number": 1,
            "source_file": Path(image_path).name,
        }],
    )
    return 1


def answer_question(question: str) -> dict:
    """
    Full query flow. Returns:
        {"answer": str, "source_image_path": str, "source_page": int, "source_file": str}
    """
    hits = query(question, k=1)
    if not hits:
        return {"answer": "Nothing has been ingested yet. Upload a document first.",
                "source_image_path": None, "source_page": None, "source_file": None}

    top_hit = hits[0]
    image_path = top_hit["metadata"]["image_path"]
    source_image = Image.open(image_path)

    answer = ask_about_image(source_image, question)

    return {
        "answer": answer,
        "source_image_path": image_path,
        "source_page": top_hit["metadata"]["page_number"],
        "source_file": top_hit["metadata"]["source_file"],
    }