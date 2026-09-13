"""A lightweight local TF-IDF retriever with no extra ML dependency."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from models.document import DocumentChunk

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'-]{1,}")


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk selected for an answer, including its similarity score."""

    chunk: DocumentChunk
    score: float


def tokenize(text: str) -> list[str]:
    """Return simple normalized word tokens appropriate for a small demo retriever."""
    return TOKEN_PATTERN.findall(text.lower())


class TfidfRetriever:
    """Ranks document chunks using cosine similarity between TF-IDF vectors."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        document_frequency = Counter(token for tokens in tokenized_chunks for token in set(tokens))
        document_count = len(chunks)
        self.idf = {
            token: math.log((1 + document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(tokens) for tokens in tokenized_chunks]
        self.norms = [self._norm(vector) for vector in self.vectors]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {token: (count / len(tokens)) * self.idf.get(token, 0.0) for token, count in counts.items()} if tokens else {}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))

    def search(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        """Return the highest-scoring relevant chunks; empty means no lexical match."""
        query_vector = self._vector(tokenize(query))
        query_norm = self._norm(query_vector)
        if not query_vector or not query_norm:
            return []
        results: list[RetrievalResult] = []
        for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms, strict=True):
            if not norm:
                continue
            dot_product = sum(query_vector.get(term, 0.0) * weight for term, weight in vector.items())
            score = dot_product / (query_norm * norm)
            if score > 0:
                results.append(RetrievalResult(chunk=chunk, score=score))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def build_retriever(chunks: list[DocumentChunk]) -> TfidfRetriever:
    """Construct a ready-to-query TF-IDF index for a processed document."""
    return TfidfRetriever(chunks)
