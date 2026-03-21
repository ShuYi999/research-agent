import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

MAX_PAGE_CHARS = 1500  # truncate long pages to avoid hitting token limits


def search_web(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the web using DuckDuckGo and return a list of results.
    Each result has: title, url, snippet.

    No API key needed — DuckDuckGo is free and open.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    return [
        {
            "title": r["title"],
            "url": r["href"],
            "snippet": r["body"],
        }
        for r in results
    ]


def read_page(url: str) -> str:
    """
    Fetch a webpage and extract its main text content.
    Returns plain text (no HTML tags), truncated to MAX_PAGE_CHARS.

    BeautifulSoup parses the HTML and pulls out readable text.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove nav, footer, scripts — we only want the main content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Truncate to avoid sending massive pages to the LLM
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + "... [truncated]"

        return text

    except Exception as e:
        return f"Error reading page: {e}"


# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────
# These tell the LLM what tools are available and what arguments they take.
# Same format as project 1 — Groq/OpenAI tool use format.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a topic. Returns a list of results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the full text content of a webpage given its URL. Use this to get detailed information from a specific page found in search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to read",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "search_web": search_web,
    "read_page": read_page,
}
