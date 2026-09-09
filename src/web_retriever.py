"""
web_retriever.py

Stage 10 of the DocuRAG pipeline: fetch supporting information from the
web for questions the uploaded PDF alone can't answer — current events,
outside comparisons, or anything requiring information beyond the document.
See query_router.py (Phase 13) for how the decision to use this gets made,
and rag_pipeline.py (Phase 14) for how its results get combined with PDF
evidence.

Web search provider decision (checked September 2026, per the project's
"verify before implementing" rule for anything involving external pricing):

- Tavily (used here): 1,000 free API credits/month, NO credit card
  required, and built specifically for AI/RAG use — clean structured
  results (title + url + content) rather than raw HTML, plus source
  filtering. Verified directly against Tavily's own docs
  (docs.tavily.com/documentation/api-credits) at the time this was written.
- Brave Search API: was a strong free option, but removed its no-card free
  tier in February 2026 — every plan now requires a credit card and meters
  usage from the first query. No longer "free" in the sense this project
  needs, so it wasn't used.
- Bing Web Search API: fully retired.
- DuckDuckGo's public API only returns basic instant answers, not general
  web search results with snippets — not usable as a real search backend.
- SerpApi: does have a free tier, but a much smaller one (100 searches/
  month), and returns raw SERP data rather than RAG-ready content.

IMPORTANT: free tiers and pricing change — Brave's removal above happened
in the same year as this project. Verify current terms at
https://tavily.com (or wherever you're reading this) before relying on
this long-term, and reconsider this choice if Tavily's terms change too.

Requires a TAVILY_API_KEY in .env (a free key is obtained at
https://tavily.com). Uses the `requests` library directly against Tavily's
REST endpoint rather than pulling in the official tavily-python SDK, since
this project avoids adding a dependency for what is a single JSON POST
request — consistent with its "no unnecessary frameworks" approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_SECONDS = 15


class MissingWebSearchAPIKeyError(Exception):
    """Raised when no Tavily API key is available."""


class WebSearchError(Exception):
    """Raised when a web search request fails outright, or returns something unusable."""


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str  # a short excerpt of the page's content, as returned by the search provider


def get_tavily_api_key() -> str | None:
    """Load TAVILY_API_KEY from a .env file or the environment. Returns None if not found."""
    load_dotenv()
    return os.getenv("TAVILY_API_KEY")


def is_configured() -> bool:
    """True if a Tavily API key is available — for the UI to check before offering web search."""
    return bool(get_tavily_api_key())


def require_tavily_api_key() -> str:
    """Like get_tavily_api_key(), but raises MissingWebSearchAPIKeyError instead of returning None."""
    api_key = get_tavily_api_key()
    if not api_key:
        raise MissingWebSearchAPIKeyError(
            "No TAVILY_API_KEY found. Add one to a .env file (see .env.example) "
            "to enable web search. Get a free key at https://tavily.com — verify "
            "current free-tier terms there before relying on it (see this "
            "module's docstring for why that matters)."
        )
    return api_key


def search_web(
    query: str,
    api_key: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_depth: str = "basic",
) -> list[WebResult]:
    """
    Run a web search and return structured results with title/url/snippet
    preserved for source attribution. A result is only ever included if the
    search provider actually returned a URL for it — a title or snippet
    with no real URL is dropped rather than shown with a fabricated link.

    Args:
        query: the search query — typically the user's question, or a
            reformulation of it (see query_router.py / rag_pipeline.py).
        api_key: if omitted, loaded via get_tavily_api_key()/.env.
        max_results: how many results to return.
        search_depth: "basic" (faster, cheaper) or "advanced" (Tavily also
            analyzes page content more thoroughly, at a higher credit
            cost). Defaults to "basic" to conserve the free tier's monthly
            credit budget — not a claim that "basic" is always sufficient
            for every query.

    Raises:
        ValueError: empty query.
        MissingWebSearchAPIKeyError: no API key available.
        WebSearchError: the request failed (network issue, invalid key,
            rate limit), or returned a response that couldn't be used.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    api_key = api_key or require_tavily_api_key()

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }

    try:
        response = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise WebSearchError(f"Web search request failed: {e}") from e

    try:
        data = response.json()
    except ValueError as e:
        raise WebSearchError(f"Web search returned an unreadable response: {e}") from e

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise WebSearchError("Web search returned an unexpected response format (no 'results' list).")

    results: list[WebResult] = []
    for item in raw_results:
        url = item.get("url")
        if not url:
            continue  # never fabricate a source; skip anything without a real URL
        results.append(WebResult(
            title=item.get("title") or url,
            url=url,
            snippet=(item.get("content") or "").strip(),
        ))
    return results


def build_web_context(results: list[WebResult]) -> str:
    """
    Join web results into LLM-ready context, each passage tagged with its
    title and URL. Mirrors retriever.build_context()'s page-tagging
    approach, but for web sources — this is what lets both the LLM (to
    cite sources) and format_web_sources() (below) know exactly which page
    each piece of external evidence came from.
    """
    if not results:
        return ""
    return "\n\n".join(f"[{r.title}]({r.url})\n{r.snippet}" for r in results)


def format_web_sources(results: list[WebResult]) -> str:
    """
    An "External Sources:" bullet list, in the format the project spec
    calls for:

        External Sources:
        - Source title (https://example.com)

    Always derived from results actually returned by the search provider —
    never invented.
    """
    if not results:
        return "No web sources were found."
    return "\n".join(f"- {r.title} ({r.url})" for r in results)


if __name__ == "__main__":
    # Quick manual test: `python src/web_retriever.py "a search query"`
    import sys

    if len(sys.argv) != 2:
        print('Usage: python web_retriever.py "<query>"')
        sys.exit(1)

    try:
        web_results = search_web(sys.argv[1])
    except (MissingWebSearchAPIKeyError, WebSearchError) as e:
        print(f"Web search failed: {e}")
        sys.exit(1)

    if not web_results:
        print("No results found.")
    else:
        for r in web_results:
            print(f"- {r.title}\n  {r.url}\n  {r.snippet[:100]}...\n")
        print(format_web_sources(web_results))
