# Insight Lens

Offline Vision Analyzer, Multimodal Edge AI Desktop App. Zero data leaves the premise.

Upload a PDF or image, then ask questions about it in plain English. Insight Lens
captions and OCRs each page locally (Florence-2), indexes the text in a local vector
store (ChromaDB), retrieves the most relevant page for your question, and answers
using a local multimodal LLM (Ollama). Nothing leaves your machine.

## Setup

1. Install [Ollama](https://ollama.com) and pull a multimodal model:
   ```
   ollama pull llava:7b
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and adjust if needed (Ollama host/model, Florence
   device).

## Run

```
streamlit run app.py
```

Upload a PDF or image from the sidebar, ingest it, then ask a question in the main
panel.

## Project structure

```
app.py                     Streamlit UI / entry point
config.py                  Central configuration, loads .env
src/
  ingestion/                PDF and image loading/preprocessing
  models/                   Florence-2 captioning/OCR + Ollama client
  rag/pipeline.py           Ingest and query orchestration
  vectorstore/               ChromaDB wrapper
data/chromadb/              Persisted vector store (gitignored)
uploads/                    Uploaded and rendered files (gitignored)
```
