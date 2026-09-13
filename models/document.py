"""Small, serializable data objects representing a processed PDF."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable passage, preserving the PDF page it came from."""

    text: str
    page_number: int
    chunk_id: int


@dataclass(frozen=True)
class ProcessedDocument:
    """Extracted document data held in the Streamlit session."""

    filename: str
    page_count: int
    character_count: int
    chunks: list[DocumentChunk]
