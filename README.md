# Research Agent

An autonomous AI research agent that searches the web, reads pages, and writes structured reports — all without human input after the initial question.

## What it does

- Accepts a research question via a chat UI or REST API
- Autonomously decides which pages to search and read
- Iterates until it has enough information to answer
- Returns a cited, structured report with sources
- Traces every step (LLM calls, tool calls, latency) in Langfuse

## How the agent loop works

Most LLM apps are single-shot: one prompt in, one response out. This agent is different — it runs in a loop where the model decides its own next action at each step:

```
User question
    ↓
LLM call — "what should I do next?"
    ↓
Tool call? → search_web(query) or read_page(url)
    ↓
Result appended to conversation history
    ↓
LLM call again — "what should I do next?"
    ↓
... repeat until LLM writes a final answer (no tool call)
```

This is the core pattern behind every production AI agent. The model drives the loop — it decides when to search, what to read, and when it has enough to answer.

## Tech stack

| Layer | Technology |
|---|---|
| LLM | qwen/qwen3-32b via [Groq](https://groq.com) |
| Web search | DuckDuckGo (via `duckduckgo-search`) |
| Page reading | BeautifulSoup (HTML scraping, script/nav stripping) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Observability | Langfuse (cloud) |

## Architecture

```
frontend.py (Streamlit)
    ↓ POST /research
main.py (FastAPI)
    ↓
ResearchAgent.run(question)
    ↓
while not done:
    Groq LLM call (with tools)
        → search_web(query)    — DuckDuckGo top-5 results
        → read_page(url)       — fetch + strip to plain text
    append result to messages
    ↓
Final answer (text response, no tool call)
    ↓
Langfuse trace (every step logged)
```

## Design decisions

**Why an agentic loop instead of a single prompt?**
A single prompt can't browse the web. The agent needs to search, evaluate what it finds, decide whether to dig deeper, and repeat. The loop enables dynamic, multi-step reasoning.

**Why DuckDuckGo instead of Google/Bing?**
No API key required — keeps the project free to run and easy to set up. In production you'd swap in a paid search API (SerpAPI, Brave Search) for better reliability.

**Why strip scripts and nav tags from pages?**
Raw HTML contains thousands of tokens of boilerplate (nav menus, JS, footers) that waste context window and add noise. BeautifulSoup strips these before passing text to the LLM.

**Why Langfuse for tracing?**
LLM apps are hard to debug without observability. Langfuse gives a full trace of every LLM call, tool call, and latency — essential for understanding why the agent made certain decisions.

## Project structure

```
research-agent/
├── main.py              # FastAPI app — /research endpoint
├── frontend.py          # Streamlit chat UI
├── requirements.txt
├── src/
│   ├── agent.py         # ResearchAgent class — agentic loop + Langfuse tracing
│   └── tools.py         # search_web and read_page tool definitions
└── tests/
    └── test_tools.py    # pytest unit tests for tools (7 tests, mocked HTTP)
```

## Getting started

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) (free tier available)
- Langfuse account (optional — tracing disabled if not configured)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file:

```
GROQ_API_KEY=your_key_here

# Optional — Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start the API

```bash
uvicorn main:app --reload
```

### 4. Start the UI

```bash
streamlit run frontend.py
```

Open `http://localhost:8501` and ask a research question.

## Example questions

- "What are the latest developments in quantum computing?"
- "What is the difference between RAG and fine-tuning?"
- "Who are the main competitors to OpenAI in 2026?"

## Running tests

```bash
pytest tests/
```

7 unit tests covering `search_web` and `read_page` — all HTTP calls are mocked so tests run offline.

## Key concepts demonstrated

- **Agentic loop** — LLM-driven decision making over multiple steps
- **Tool use** — LLM calls external functions and incorporates results
- **Web scraping** — HTML fetching and parsing with BeautifulSoup
- **Observability** — full request tracing with Langfuse (cloud)
- **Unit testing** — mocking external HTTP calls with pytest + unittest.mock
