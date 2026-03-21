import json
import os

from groq import Groq

from src.tools import TOOLS, AVAILABLE_FUNCTIONS

CHAT_MODEL = "qwen/qwen3-32b"
MAX_ITERATIONS = 10  # prevent infinite loops

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a research agent. Your job is to answer the user's research question thoroughly and accurately.

Use the available tools to search the web and read pages. Always gather information before answering.

Rules:
- Always use tools to search before answering — never rely on training data alone
- Cite your sources (include URLs in your final answer)
- When you have enough information, write a clear, well-structured final report
"""


# ── LANGFUSE SETUP ─────────────────────────────────────────────────────────
try:
    from langfuse import Langfuse

    _langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    _TRACING = True
    print("Langfuse tracing enabled.")
except Exception:
    _langfuse = None
    _TRACING = False


class ResearchAgent:
    """
    An autonomous research agent that uses web search and page reading
    to answer questions. It runs in a loop, deciding its own next action
    at each step, until it has enough information to write a final report.
    """

    def __init__(self):
        pass

    def run(self, question: str) -> dict:
        """
        Run the agent on a research question.

        Returns a dict with:
        - answer: the final research report
        - steps: list of actions taken (for display in the UI)
        - iterations: how many LLM calls were made
        """
        # Start Langfuse trace
        trace = (
            _langfuse.trace(name="research-agent", input=question) if _TRACING else None
        )

        # Conversation history — grows with each step
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research question: {question}"},
        ]

        steps = []  # track what the agent did (for UI display)
        iterations = 0

        # ── AGENT LOOP ────────────────────────────────────────────────────
        # Each iteration: ask LLM what to do next → execute tool → repeat
        # Loop ends when: LLM gives a text answer (no tool call) OR max iterations reached

        while iterations < MAX_ITERATIONS:
            iterations += 1

            # Log this LLM call as a generation in Langfuse
            gen = (
                trace.generation(
                    name=f"step-{iterations}",
                    model=CHAT_MODEL,
                    input=messages,
                )
                if trace
                else None
            )

            # Ask the LLM: what should I do next?
            response = _groq.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if gen:
                gen.end(output=msg.content or str(msg.tool_calls))

            # ── CASE 1: LLM wants to use a tool ───────────────────────────
            if msg.tool_calls:
                # Add assistant message to history
                messages.append(self._msg_to_dict(msg))

                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    # Log the step for UI display
                    steps.append(
                        {
                            "tool": fn_name,
                            "args": fn_args,
                        }
                    )
                    print(f"[Agent] Step {iterations}: {fn_name}({fn_args})")

                    # Execute the tool on our machine
                    tool_span = (
                        trace.span(name=fn_name, input=fn_args) if trace else None
                    )

                    try:
                        result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)
                    except Exception as e:
                        result = f"Tool error: {e}"

                    if tool_span:
                        tool_span.end(output=result)

                    # Add tool result to history so LLM sees it next iteration
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

            # ── CASE 2: LLM is done — final answer ────────────────────────
            else:
                final_answer = msg.content or "No answer generated."
                print(f"[Agent] Done after {iterations} iterations.")

                if trace:
                    trace.update(output=final_answer)
                    try:
                        _langfuse.flush()
                    except Exception:
                        pass

                return {
                    "answer": final_answer,
                    "steps": steps,
                    "iterations": iterations,
                }

        # ── MAX ITERATIONS REACHED ─────────────────────────────────────────
        # Agent didn't finish in time — return what we have
        fallback = "Research incomplete: maximum iterations reached. Please try a more specific question."
        if trace:
            trace.update(output=fallback)
            try:
                _langfuse.flush()
            except Exception:
                pass

        return {
            "answer": fallback,
            "steps": steps,
            "iterations": iterations,
        }

    def _msg_to_dict(self, msg) -> dict:
        """Convert a Groq response message object to a plain dict for history."""
        d = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
