import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Research Agent", page_icon="🔍", layout="centered")

st.title("🔍 Research Agent")
st.caption(
    "Ask any research question — the agent will search the web and compile a report."
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. You ask a research question
    2. The agent searches the web
    3. It reads relevant pages
    4. It compiles a structured report

    **Example questions:**
    - What is the current state of AI agents?
    - How does RAG compare to fine-tuning?
    - What are the best practices for LLM evaluation?
    - What is ChromaDB and how does it work?
    """)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ── RESEARCH INPUT ────────────────────────────────────────────────────────────
question = st.text_input(
    "Research question",
    placeholder="e.g. What is the current state of AI agents in 2025?",
)

if st.button("🔍 Research", type="primary", disabled=not question.strip()) or (
    question.strip() and question != st.session_state.get("last_question")
):
    with st.spinner("Agent is researching... this may take a minute."):
        try:
            response = requests.post(
                f"{API_URL}/research",
                json={"question": question},
                timeout=300,
            )
            st.session_state.result = response.json()
            st.session_state.last_question = question
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            st.stop()

# ── SHOW RESULTS ──────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    if result.get("steps"):
        with st.expander(
            f"🔎 Agent steps ({result['iterations']} iterations)", expanded=True
        ):
            for i, step in enumerate(result["steps"], 1):
                tool = step["tool"]
                args = step["args"]
                if tool == "search_web":
                    st.markdown(f"**Step {i}:** Searched for `{args.get('query')}`")
                elif tool == "read_page":
                    url = args.get("url", "")
                    st.markdown(f"**Step {i}:** Read page [{url[:60]}...]({url})")

    st.divider()
    st.markdown("## Research Report")
    st.markdown(result.get("answer", "No answer returned."))
