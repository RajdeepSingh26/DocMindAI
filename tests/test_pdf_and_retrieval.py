from models.document import DocumentChunk
from services.pdf_service import _chunk_page_text
from services.retrieval_service import build_retriever


def test_chunking_preserves_page_number() -> None:
    chunks = _chunk_page_text("Important policy details. " * 100, page_number=4, first_chunk_id=0)
    assert len(chunks) > 1
    assert all(chunk.page_number == 4 for chunk in chunks)


def test_retrieval_returns_best_matching_chunk() -> None:
    chunks = [
        DocumentChunk("The renewal deadline is 31 March 2026.", page_number=2, chunk_id=0),
        DocumentChunk("The office opens at nine o'clock.", page_number=5, chunk_id=1),
    ]
    result = build_retriever(chunks).search("When is the renewal deadline?", top_k=1)
    assert result[0].chunk.page_number == 2
