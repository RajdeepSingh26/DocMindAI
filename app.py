"""Streamlit entry point for AI Document Assistant."""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from models.document import ProcessedDocument
from services.ai_service import AIServiceError, answer_question, provider_status
from services.pdf_service import PDFExtractionError, extract_pdf
from services.retrieval_service import build_retriever

load_dotenv()

APP_TITLE = "AI Document Assistant"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def initialise_state() -> None:
    """Create the session values used to retain a document and its chat."""
    st.session_state.setdefault("document", None)
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("document_hash", None)
    st.session_state.setdefault("messages", [])


def reset_document() -> None:
    """Clear all document-specific state."""
    for key in ("document", "retriever", "document_hash", "messages"):
        st.session_state[key] = None if key != "messages" else []


def process_upload(uploaded_file: Any) -> None:
    """Extract the uploaded PDF and prepare the in-memory retrieval index."""
    data = uploaded_file.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    if file_hash == st.session_state.document_hash:
        return
    if len(data) > MAX_UPLOAD_BYTES:
        raise PDFExtractionError("This PDF is larger than the 15 MB demo limit.")

    document = extract_pdf(data, uploaded_file.name)
    st.session_state.document = document
    st.session_state.retriever = build_retriever(document.chunks)
    st.session_state.document_hash = file_hash
    st.session_state.messages = []


def document_sidebar() -> None:
    """Render upload controls and document metadata in the sidebar."""
    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            try:
                with st.spinner("Extracting and indexing the PDF..."):
                    process_upload(uploaded_file)
                st.success("Document ready")
            except PDFExtractionError as error:
                reset_document()
                st.error(str(error))

        document: ProcessedDocument | None = st.session_state.document
        if document:
            st.divider()
            st.caption("DOCUMENT INFORMATION")
            st.write(f"**File:** {document.filename}")
            st.write(f"**Pages:** {document.page_count}")
            st.write(f"**Extracted characters:** {document.character_count:,}")
            st.write(f"**Searchable chunks:** {len(document.chunks)}")
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            if st.button("Upload another document", use_container_width=True):
                reset_document()
                st.rerun()


def render_chat() -> None:
    """Render conversation history and handle a new grounded question."""
    document: ProcessedDocument | None = st.session_state.document
    if not document:
        st.info("Upload a text-based PDF from the sidebar to begin.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("pages"):
                st.caption("Sources: " + ", ".join(f"page {page}" for page in message["pages"]))

    question = st.chat_input("Ask a question about this document")
    if question is None:
        return
    question = question.strip()
    if not question:
        st.warning("Please enter a question before sending it.")
        return
    ready, message = provider_status()
    if not ready:
        st.error(message)
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding relevant passages and generating an answer..."):
            try:
                results = st.session_state.retriever.search(question, top_k=4)
                answer = answer_question(
                    question=question,
                    results=results,
                    chat_history=st.session_state.messages[:-1],
                )
                st.markdown(answer)
                pages = sorted({result.chunk.page_number for result in results})
                if pages:
                    st.caption("Sources: " + ", ".join(f"page {page}" for page in pages))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "pages": pages}
                )
            except AIServiceError as error:
                st.error(str(error))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    initialise_state()
    document_sidebar()

    st.title("📄 AI Document Assistant")
    st.markdown(
        "Upload a PDF and ask questions in plain language. The app searches relevant "
        "passages locally, then asks your selected AI provider to answer only from those passages."
    )
    render_chat()


if __name__ == "__main__":
    main()
