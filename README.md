# AI Document Assistant

AI Document Assistant is a Streamlit portfolio project that lets a user upload a text-based PDF and ask natural-language questions about it. It extracts the document locally, retrieves the most relevant passages using local TF-IDF cosine similarity, and sends only that bounded context to a local Ollama model by default.

It is deliberately a lightweight retrieval-augmented generation (RAG) implementation: there is local retrieval, but no vector database, embeddings API, OCR, or hosted file search.

## Features

- PDF upload with filename, page count, extracted character count, and chunk count
- Page-aware text extraction with PyPDF
- Overlapping text chunking for large documents
- Local TF-IDF/cosine-similarity retrieval, with no additional ML package
- Grounded local Ollama answers that explicitly decline when the retrieved text does not support an answer
- Source page numbers beside every answer
- Multi-turn chat history for a single uploaded document
- Clear-conversation and upload-another-document controls
- Friendly handling of invalid, encrypted, scanned/image-only, and oversized PDFs

## Architecture

```text
User
  ↓
Streamlit UI
  ↓
PDF Upload
  ↓
PDF Text Extraction
  ↓
Text Chunking
  ↓
Local Retrieval
  ↓
Relevant Context
  ↓
LLM API
  ↓
Answer + Sources
  ↓
User
```

`services/pdf_service.py` extracts page text and chunks it. `services/retrieval_service.py` creates a simple in-memory TF-IDF index and ranks chunks with cosine similarity. `services/ai_service.py` builds a bounded, page-labelled context and calls the OpenAI Responses API. `app.py` owns Streamlit state and presentation.

## Technologies

- Python 3.13+
- Streamlit
- Ollama local HTTP API (free, local default)
- OpenAI Python SDK and Responses API (optional paid provider)
- PyPDF
- python-dotenv

## Project structure

```text
ai_document_assistant/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── models/
│   ├── __init__.py
│   └── document.py
├── services/
│   ├── __init__.py
│   ├── ai_service.py
│   ├── pdf_service.py
│   └── retrieval_service.py
├── tests/
│   └── test_pdf_and_retrieval.py
└── utils/
    └── __init__.py
```

## Installation

1. Install Python 3.13 or newer.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Install [Ollama](https://ollama.com/download), then download the free local model:

   ```powershell
   ollama pull llama3.2:3b
   ```

5. Copy `.env.example` to `.env`. The default configuration is free and uses local Ollama; it needs no API key. Keep `.env` private; it is ignored by Git.

   ```text
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2:3b
   ```

6. Run the application:

   ```powershell
   streamlit run app.py
   ```

For an optional paid hosted alternative, set `LLM_PROVIDER=openai`, then add `OPENAI_API_KEY` and `OPENAI_MODEL` to `.env`.

## Example usage

Upload a text-based annual report, policy, or research paper. Then ask: “What are the main risks described in this document?” or “What deadline is mentioned for the application?” The answer includes the retrieved source pages. Use **Clear conversation** to reset the chat while keeping the same PDF.

## Limitations

- Image-only/scanned PDFs are not supported because this version has no OCR.
- Retrieval is lexical TF-IDF, so it can miss a semantic match that uses very different wording.
- The demo limits uploads to 15 MB, 250 pages, and 1.5 million extracted characters.
- A model can only answer from the retrieved passages, not every word in a huge document at once.

## Possible future improvements

- Add OCR for scanned documents.
- Use embeddings and a vector database for stronger semantic retrieval.
- Add citations that link to a rendered PDF page.
- Stream model output and add exportable chat transcripts.
- Add automated evaluation cases for answer grounding and retrieval quality.
