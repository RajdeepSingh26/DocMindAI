"""PDF validation, text extraction, and page-aware chunking."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from models.document import DocumentChunk, ProcessedDocument

MAX_PAGES = 250
MAX_EXTRACTED_CHARACTERS = 1_500_000
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200


class PDFExtractionError(Exception):
    """A user-safe error raised when a PDF cannot be prepared."""


def _chunk_page_text(text: str, page_number: int, first_chunk_id: int) -> list[DocumentChunk]:
    """Split one page into overlapping, readable chunks without losing page provenance."""
    normalized = " ".join(text.split())
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_id = first_chunk_id
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary
        passage = normalized[start:end].strip()
        if passage:
            chunks.append(DocumentChunk(text=passage, page_number=page_number, chunk_id=chunk_id))
            chunk_id += 1
        if end >= len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def extract_pdf(pdf_bytes: bytes, filename: str) -> ProcessedDocument:
    """Extract text from a PDF and return chunks suitable for local retrieval."""
    if not pdf_bytes or not filename.lower().endswith(".pdf"):
        raise PDFExtractionError("Please upload a valid PDF file.")
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except PdfReadError as error:
        raise PDFExtractionError("This file is not a readable PDF or is corrupted.") from error
    except Exception as error:
        raise PDFExtractionError("The PDF could not be opened.") from error

    if reader.is_encrypted:
        raise PDFExtractionError("Password-protected PDFs are not supported.")
    if not reader.pages:
        raise PDFExtractionError("This PDF has no pages to read.")
    if len(reader.pages) > MAX_PAGES:
        raise PDFExtractionError(f"This PDF has more than the {MAX_PAGES}-page demo limit.")

    chunks: list[DocumentChunk] = []
    extracted_characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise PDFExtractionError(f"Text extraction failed on page {page_number}.") from error
        extracted_characters += len(text)
        if extracted_characters > MAX_EXTRACTED_CHARACTERS:
            raise PDFExtractionError("This PDF contains too much text for the demo limit.")
        page_chunks = _chunk_page_text(text, page_number, len(chunks))
        chunks.extend(page_chunks)

    if not chunks:
        raise PDFExtractionError(
            "No readable text was found. This may be a scanned or image-only PDF; OCR is not included."
        )
    return ProcessedDocument(
        filename=filename,
        page_count=len(reader.pages),
        character_count=extracted_characters,
        chunks=chunks,
    )
