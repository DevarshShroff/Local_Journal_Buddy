"""
brain.py — Module C: The Brain (Ollama Integration)
Local_Journal_Buddy · Llama3:8b via Ollama · 100% offline

Responsibilities:
  1. Receive a user question
  2. Pull top-k relevant chunks from Module B (librarian.query)
  3. Build a structured prompt (system + context + question)
  4. Stream the response from Ollama's local REST API
  5. Graceful fallback — if Ollama is unreachable, return retrieved
     journal chunks as plain readable text

Public API (consumed by Module D / Tauri backend):
    ask(question)           -> BrainResponse
    ask_stream(question)    -> Iterator[str]   (token-by-token streaming)
    health_check()          -> OllamaStatus

Usage (CLI):
    python brain.py --ask "when did I last feel really proud of myself"
    python brain.py --ask "what patterns do you see in my stress" --stream
    python brain.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger("brain")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3:8b"
OLLAMA_TIMEOUT   = 120          # seconds — generous for first cold-start load
TOP_K_CHUNKS     = 4            # how many journal chunks to retrieve per query
MAX_CONTEXT_CHARS = 3000        # cap total context fed to LLM to avoid overflow

SYSTEM_PROMPT = """You are an empathetic, private mental health assistant named Sage.
You have access to excerpts from the user's private handwritten journal entries.

Your goals:
- Help the user identify emotional patterns across time
- Celebrate their wins and moments of resilience
- Reflect their own words back to them with compassion
- Ask one gentle follow-up question to deepen reflection when appropriate

Your rules:
- NEVER suggest or imply the data is being sent anywhere — it is 100% private and local
- NEVER diagnose, prescribe, or replace professional mental health care
- If the journal context is insufficient, say so honestly rather than fabricating
- Keep responses warm, concise (3–5 sentences), and grounded in the actual journal excerpts
- Refer to journal entries naturally: "In one of your entries..." or "You wrote about..."
"""


# ── Data contracts ────────────────────────────────────────────────────────────
@dataclass
class OllamaStatus:
    reachable:      bool
    model_available: bool
    model:          str
    error:          Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BrainResponse:
    question:       str
    answer:         str
    chunks_used:    int
    fallback_used:  bool          # True if Ollama was unreachable
    context_dates:  list[str]     # dates of journal entries used
    errors:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── Lazy import guard ─────────────────────────────────────────────────────────
def _require_requests():
    try:
        import requests
        return requests
    except ImportError:
        sys.exit(
            "[brain] Missing: 'requests'\n"
            "Install: pip install requests"
        )


# ── Ollama health check ───────────────────────────────────────────────────────
def health_check() -> OllamaStatus:
    """Ping Ollama and confirm the required model is available."""
    requests = _require_requests()

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_available = any(OLLAMA_MODEL in m for m in models)
        return OllamaStatus(
            reachable=True,
            model_available=model_available,
            model=OLLAMA_MODEL,
            error=None if model_available else (
                f"Model '{OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return OllamaStatus(
            reachable=False,
            model_available=False,
            model=OLLAMA_MODEL,
            error=str(exc),
        )


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_context_block(chunks) -> tuple[str, list[str]]:
    """
    Format retrieved journal chunks into a context block for the prompt.
    Returns (context_string, list_of_dates_used).
    Truncates to MAX_CONTEXT_CHARS to avoid overflowing the context window.
    """
    if not chunks:
        return "No relevant journal entries found.", []

    lines: list[str] = ["RELEVANT JOURNAL EXCERPTS (most similar to the question):"]
    dates_used: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        entry = (
            f"\n[Entry {i} — {chunk.date} "
            f"(relevance: {1 - chunk.distance:.0%})]\n"
            f"{chunk.text}"
        )
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            logger.info("Context truncated at chunk %d to stay within limit.", i - 1)
            break
        lines.append(entry)
        dates_used.append(chunk.date)
        total_chars += len(entry)

    return "\n".join(lines), dates_used


def _build_prompt(question: str, context_block: str) -> str:
    return (
        f"{context_block}\n\n"
        f"─────────────────────────────────────────\n"
        f"USER QUESTION: {question}\n\n"
        f"Respond as Sage, grounding your answer in the journal excerpts above."
    )


# ── Fallback response (Ollama unreachable) ────────────────────────────────────
def _fallback_response(question: str, chunks, error: str) -> BrainResponse:
    """
    When Ollama is down, surface the raw journal chunks in a readable format
    so the app remains useful.
    """
    if not chunks:
        answer = (
            "⚠ Ollama is not running and no relevant journal entries were found.\n\n"
            f"To start Ollama: open a terminal and run `ollama serve`\n"
            f"Error: {error}"
        )
        return BrainResponse(
            question=question, answer=answer,
            chunks_used=0, fallback_used=True,
            context_dates=[], errors=[error],
        )

    lines = [
        "⚠ Ollama is not running — showing your raw journal excerpts instead.\n",
        f"Here are the most relevant entries for: \"{question}\"\n",
        "─" * 50,
    ]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"\n[{chunk.date}]\n{chunk.text}")

    lines += [
        "\n" + "─" * 50,
        "\nTo get an AI-generated response, start Ollama:",
        "  1. Open Terminal",
        f"  2. Run: ollama serve",
        f"  3. Run: ollama pull {OLLAMA_MODEL}  (first time only)",
        f"\nError detail: {error}",
    ]

    return BrainResponse(
        question=question,
        answer="\n".join(lines),
        chunks_used=len(chunks),
        fallback_used=True,
        context_dates=[c.date for c in chunks],
        errors=[error],
    )


# ── Ollama call ───────────────────────────────────────────────────────────────
def _call_ollama(prompt: str) -> str:
    """Single-shot (non-streaming) Ollama call. Returns full response text."""
    requests = _require_requests()

    payload = {
        "model":  OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,      # warm but not hallucination-prone
            "top_p": 0.9,
            "num_predict": 512,      # max tokens in response
        },
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _stream_ollama(prompt: str) -> Iterator[str]:
    """Token-by-token streaming from Ollama. Yields string tokens."""
    requests = _require_requests()

    payload = {
        "model":  OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=OLLAMA_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break


# ── Public API ─────────────────────────────────────────────────────────────────
def ask(question: str, top_k: int = TOP_K_CHUNKS) -> BrainResponse:
    """
    Main entry point for Module D / Tauri backend.

    Flow: question → librarian.query → prompt builder → Ollama → BrainResponse
    Falls back gracefully if Ollama is unreachable.

    Usage:
        from brain import ask
        response = ask("when did I last feel proud of myself?")
        print(response.answer)
    """
    # Step 1 — retrieve relevant journal chunks
    try:
        from librarian import query as db_query
        chunks = db_query(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.error("Librarian query failed: %s", exc)
        chunks = []

    # Step 2 — build prompt
    context_block, dates_used = _build_context_block(chunks)
    prompt = _build_prompt(question, context_block)

    # Step 3 — check Ollama, call or fallback
    status = health_check()

    if not status.reachable or not status.model_available:
        error = status.error or "Ollama unreachable"
        logger.warning("Ollama unavailable (%s) — using fallback.", error)
        return _fallback_response(question, chunks, error)

    try:
        logger.info("Sending prompt to Ollama (%s)…", OLLAMA_MODEL)
        answer = _call_ollama(prompt)
        return BrainResponse(
            question=question,
            answer=answer,
            chunks_used=len(chunks),
            fallback_used=False,
            context_dates=dates_used,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama call failed: %s", exc)
        return _fallback_response(question, chunks, str(exc))


def ask_stream(question: str, top_k: int = TOP_K_CHUNKS) -> Iterator[str]:
    """
    Streaming version — yields tokens as they arrive from Ollama.
    Falls back to yielding the fallback response as a single string.

    Usage (Module D / websocket):
        for token in ask_stream("what patterns do you see in my stress?"):
            send_to_ui(token)
    """
    try:
        from librarian import query as db_query
        chunks = db_query(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        chunks = []
        logger.error("Librarian query failed: %s", exc)

    context_block, _ = _build_context_block(chunks)
    prompt = _build_prompt(question, context_block)

    status = health_check()
    if not status.reachable or not status.model_available:
        fallback = _fallback_response(question, chunks, status.error or "Ollama unreachable")
        yield fallback.answer
        return

    try:
        yield from _stream_ollama(prompt)
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_response(question, chunks, str(exc))
        yield fallback.answer


# ── CLI ────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain",
        description="Local_Journal_Buddy — Module C: The Brain (Ollama)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ask",    metavar="QUESTION", help="Ask a question about your journals")
    g.add_argument("--status", action="store_true", help="Check if Ollama is running")

    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    p.add_argument("--top-k",  metavar="N", type=int, default=TOP_K_CHUNKS,
                   help=f"Journal chunks to retrieve (default: {TOP_K_CHUNKS})")
    p.add_argument("--json",   action="store_true", help="Output BrainResponse as JSON")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.status:
        status = health_check()
        if args.json:
            print(json.dumps(status.to_dict(), indent=2))
        else:
            icon = "✓" if status.reachable else "✗"
            print(f"{icon} Ollama reachable : {status.reachable}")
            print(f"{icon} Model available  : {status.model_available}  ({status.model})")
            if status.error:
                print(f"  → {status.error}")
        return

    if args.stream:
        for token in ask_stream(args.ask, top_k=args.top_k):
            print(token, end="", flush=True)
        print()  # newline at end
    else:
        response = ask(args.ask, top_k=args.top_k)
        if args.json:
            print(response.to_json())
        else:
            print(f"\n{'─'*60}")
            print(f"Q: {response.question}")
            print(f"{'─'*60}")
            print(response.answer)
            print(f"{'─'*60}")
            src = "FALLBACK" if response.fallback_used else OLLAMA_MODEL
            print(f"Source: {src} · Chunks used: {response.chunks_used} · Dates: {', '.join(response.context_dates) or 'none'}")


if __name__ == "__main__":
    main()