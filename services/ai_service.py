"""LLM provider integration for grounded document answers.

Ollama is the default provider so this demo can run locally without API credits.
OpenAI remains an optional provider for users who want to use the hosted API.
"""

from __future__ import annotations

import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from services.retrieval_service import RetrievalResult

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_CONTEXT_CHARACTERS = 14_000


class AIServiceError(Exception):
    """A user-safe error raised for configuration and OpenAI API failures."""


def _format_context(results: list[RetrievalResult]) -> str:
    """Build a bounded prompt section with explicit page citations."""
    passages: list[str] = []
    current_length = 0
    for result in results:
        passage = f"[Page {result.chunk.page_number}]\n{result.chunk.text}"
        if current_length + len(passage) > MAX_CONTEXT_CHARACTERS:
            break
        passages.append(passage)
        current_length += len(passage)
    return "\n\n---\n\n".join(passages)


def _recent_history(chat_history: list[dict[str, object]]) -> str:
    """Keep a small amount of prior conversation without letting it override sources."""
    lines = []
    for message in chat_history[-6:]:
        role = str(message.get("role", "user")).capitalize()
        content = str(message.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No previous conversation."


def _provider() -> str:
    """Return the configured provider name, defaulting to free local Ollama."""
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def provider_status() -> tuple[bool, str]:
    """Validate configuration before a user submits a question."""
    provider = _provider()
    if provider == "ollama":
        return True, ""
    if provider == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return True, ""
        return False, "OPENAI_API_KEY is missing. Add it to .env and restart Streamlit."
    return False, "LLM_PROVIDER must be either 'ollama' or 'openai'."


def _build_prompt(question: str, context: str, chat_history: list[dict[str, object]]) -> str:
    """Create a provider-neutral grounded prompt."""
    return (
        "You are a precise document question-answering assistant. Answer only using the "
        "provided document passages. Do not use outside knowledge or make inferences that "
        "are not supported by the passages. If the answer is absent or uncertain, say exactly: "
        "'I cannot find that information in the uploaded document.' Keep the answer concise. "
        "When useful, cite supporting pages in brackets, for example [Page 3].\n\n"
        f"Previous conversation (for conversational context only):\n{_recent_history(chat_history)}\n\n"
        f"Document passages:\n{context}\n\nQuestion: {question}"
    )


def _answer_with_ollama(prompt: str) -> str:
    """Call a locally running Ollama model through its HTTP API."""
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    payload = json.dumps(
        {"model": os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL), "prompt": prompt, "stream": False}
    ).encode("utf-8")
    request = Request(f"{base_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=120) as response:  # nosec B310: local configurable Ollama endpoint
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise AIServiceError("Ollama rejected the request. Check that the configured model is installed.") from error
    except URLError as error:
        raise AIServiceError(
            "Could not reach Ollama. Install Ollama, run 'ollama serve', and download the configured model."
        ) from error
    except (json.JSONDecodeError, TimeoutError) as error:
        raise AIServiceError("Ollama returned an invalid or timed-out response. Please try again.") from error
    answer = str(data.get("response", "")).strip()
    if not answer:
        raise AIServiceError("Ollama returned an empty response. Please try again.")
    return answer


def answer_question(
    question: str, results: list[RetrievalResult], chat_history: list[dict[str, object]],
) -> str:
    """Ask OpenAI to answer solely from local retrieval results."""
    context = _format_context(results)
    if not context:
        return "I cannot find that information in the uploaded document."

    prompt = _build_prompt(question, context, chat_history)
    provider = _provider()
    if provider == "ollama":
        return _answer_with_ollama(prompt)
    if provider != "openai":
        raise AIServiceError("LLM_PROVIDER must be either 'ollama' or 'openai'.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AIServiceError("OPENAI_API_KEY is not configured.")
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            instructions="Answer only from the provided document passages. If unsupported, say you cannot find it.",
            input=prompt,
            max_output_tokens=600,
            store=False,
        )
    except APIConnectionError as error:
        raise AIServiceError("Could not reach OpenAI. Check your internet connection and try again.") from error
    except APIStatusError as error:
        raise AIServiceError(f"OpenAI returned an error ({error.status_code}). Check your API key and model access.") from error
    except APIError as error:
        raise AIServiceError("OpenAI could not complete this request. Please try again.") from error
    except Exception as error:
        raise AIServiceError("The AI request could not be completed. Check your configuration and try again.") from error

    answer = (response.output_text or "").strip()
    if not answer:
        raise AIServiceError("OpenAI returned an empty response. Please try again.")
    return answer
