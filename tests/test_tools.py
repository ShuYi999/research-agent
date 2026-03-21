"""
Tests for research-agent tools.py

We use mocking to avoid hitting the real internet during tests.
unittest.mock is built into Python — no extra install needed.
"""

from unittest.mock import MagicMock, patch

from src.tools import MAX_PAGE_CHARS, read_page, search_web


# ── search_web tests ──────────────────────────────────────────────────────────


def test_search_web_returns_list():
    """search_web should always return a list."""
    fake_results = [
        {
            "title": "Python docs",
            "href": "https://python.org",
            "body": "Official Python docs",
        },
        {
            "title": "Real Python",
            "href": "https://realpython.com",
            "body": "Python tutorials",
        },
    ]

    # patch() temporarily replaces DDGS with a fake version for this test only
    with patch("src.tools.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results
        results = search_web("python")

    assert isinstance(results, list)


def test_search_web_result_has_correct_keys():
    """Each result should have title, url, and snippet keys."""
    fake_results = [
        {
            "title": "Python docs",
            "href": "https://python.org",
            "body": "Official Python docs",
        },
    ]

    with patch("src.tools.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results
        results = search_web("python")

    assert "title" in results[0]
    assert "url" in results[0]
    assert "snippet" in results[0]


def test_search_web_maps_href_to_url():
    """DuckDuckGo returns 'href' but our function should rename it to 'url'."""
    fake_results = [
        {"title": "Test", "href": "https://example.com", "body": "Some text"},
    ]

    with patch("src.tools.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results
        results = search_web("test")

    # href from DuckDuckGo should become url in our output
    assert results[0]["url"] == "https://example.com"


# ── read_page tests ───────────────────────────────────────────────────────────


def test_read_page_returns_text():
    """read_page should return the visible text from a webpage."""
    fake_html = "<html><body><p>Hello world</p></body></html>"

    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value.text = fake_html
        mock_get.return_value.raise_for_status = MagicMock()
        result = read_page("https://example.com")

    assert "Hello world" in result


def test_read_page_strips_scripts_and_nav():
    """Scripts, nav, and footer tags should be removed from the output."""
    fake_html = """
    <html><body>
        <nav>Menu items</nav>
        <script>alert('hi')</script>
        <p>Real content here</p>
        <footer>Footer text</footer>
    </body></html>
    """

    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value.text = fake_html
        mock_get.return_value.raise_for_status = MagicMock()
        result = read_page("https://example.com")

    assert "Real content here" in result
    assert "Menu items" not in result
    assert "alert" not in result
    assert "Footer text" not in result


def test_read_page_truncates_long_content():
    """Pages longer than MAX_PAGE_CHARS should be truncated."""
    long_text = "a" * (MAX_PAGE_CHARS + 500)
    fake_html = f"<html><body><p>{long_text}</p></body></html>"

    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value.text = fake_html
        mock_get.return_value.raise_for_status = MagicMock()
        result = read_page("https://example.com")

    assert len(result) <= MAX_PAGE_CHARS + len("... [truncated]")
    assert result.endswith("... [truncated]")


def test_read_page_handles_network_error():
    """If the request fails, read_page should return an error string, not crash."""
    with patch("src.tools.requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        result = read_page("https://example.com")

    assert "Error reading page" in result
