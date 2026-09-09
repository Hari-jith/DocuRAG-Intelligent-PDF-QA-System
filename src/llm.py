"""
llm.py

Stage 7 of the DocuRAG pipeline (generation): a thin wrapper around the
Gemini API for sending a (system_prompt, user_prompt) pair and getting back
generated text.

This module deliberately knows nothing about RAG, retrieval, or grounding
rules — it only knows how to talk to Gemini. The actual grounding
instructions (e.g. "answer only from the given PDF context, say so if the
answer isn't there") live in rag_pipeline.py (Phase 14) and summarizer.py
(Phase 10) — the modules that know *what* to ask for. Keeping generate()
generic means it can be reused for grounded Q&A, summarization, and
anything else needing an LLM call, without duplicating API-handling code
in every one of those modules.

Uses the current `google-genai` SDK. The older `google-generativeai`
package has been fully deprecated by Google and should not be used.

Depends on: nothing else in src/. This module is intentionally standalone
so summarizer.py and rag_pipeline.py can both depend on it without any
circular relationship between modules.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types


# Once config.py exists, these two defaults should move there alongside
# CHUNK_SIZE, TOP_K, and EMBEDDING_MODEL rather than staying local constants.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_TEMPERATURE = 0.2  # low by default: this project prioritizes grounded, factual answers over varied/creative ones


class MissingAPIKeyError(Exception):
    """Raised when no Gemini API key is available (not passed in, and not found in the environment/.env file)."""


class LLMGenerationError(Exception):
    """Raised when a Gemini API call fails outright, or returns a response with no usable text."""


def get_gemini_api_key() -> str | None:
    """
    Load GEMINI_API_KEY from a .env file (via python-dotenv) or the
    environment. Returns None rather than raising if not found — callers
    decide whether that's fatal (require_gemini_api_key()) or just means
    "skip this feature for now" (e.g. the experiments notebook, which runs
    fine without a key and just shows the prompt instead of a real answer).
    """
    load_dotenv()
    return os.getenv("GEMINI_API_KEY")


def is_configured() -> bool:
    """
    True if a Gemini API key is available. Meant for the Streamlit UI
    (Phase 11) to show a clear setup message before the user even asks a
    question, rather than letting them hit a confusing error mid-chat.
    """
    return bool(get_gemini_api_key())


def require_gemini_api_key() -> str:
    """
    Like get_gemini_api_key(), but raises MissingAPIKeyError instead of
    returning None. Used right before an LLM call actually needs to happen,
    so a missing key fails clearly and immediately rather than as a
    confusing downstream API error.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise MissingAPIKeyError(
            "No GEMINI_API_KEY found. Add it to a .env file in the project "
            "root (see .env.example) to enable LLM features. Get a key at "
            "https://ai.google.dev — verify current free-tier availability "
            "there before relying on it, since this can change over time."
        )
    return api_key


def generate(
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int | None = None,
) -> str:
    """
    Send one prompt to Gemini and return the generated text.

    Args:
        system_prompt: grounding/behavior instructions — what the model
            should and shouldn't do. Kept separate from user_prompt so the
            same rules can be reused across many calls without repeating
            them inline.
        user_prompt: the actual request for this specific call (e.g.
            retrieved PDF context plus a question, or a chunk of text to
            summarize).
        api_key: if omitted, loaded via get_gemini_api_key()/.env.
        model: the Gemini model name to use.
        temperature: lower = more deterministic/conservative, higher = more
            varied. See the DEFAULT_TEMPERATURE note above for why this
            defaults low here.
        max_output_tokens: optional cap on response length.

    Returns:
        The generated text, stripped of leading/trailing whitespace.

    Raises:
        MissingAPIKeyError: no API key was provided or found.
        LLMGenerationError: the API call failed (network issue, invalid
            key, rate limit, etc.), or returned a response with no usable
            text (a malformed/empty response).
    """
    api_key = api_key or require_gemini_api_key()

    config_kwargs: dict = {"system_instruction": system_prompt, "temperature": temperature}
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = max_output_tokens

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as e:
        raise LLMGenerationError(f"Gemini API call failed: {e}") from e

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise LLMGenerationError("Gemini returned an empty or malformed response.")

    return text.strip()


if __name__ == "__main__":
    # Quick manual test: `python src/llm.py "<system prompt>" "<user prompt>"`
    import sys

    if len(sys.argv) != 3:
        print('Usage: python llm.py "<system_prompt>" "<user_prompt>"')
        sys.exit(1)

    if not is_configured():
        print(
            "No GEMINI_API_KEY found in the environment — nothing to call. "
            "Add one to a .env file (see .env.example) and try again."
        )
        sys.exit(1)

    try:
        answer = generate(system_prompt=sys.argv[1], user_prompt=sys.argv[2])
    except (MissingAPIKeyError, LLMGenerationError) as e:
        print(f"LLM call failed: {e}")
        sys.exit(1)

    print(answer)
