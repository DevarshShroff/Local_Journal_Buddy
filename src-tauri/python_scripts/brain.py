from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

DEFAULT_MODEL = (
    os.environ.get("JOURNAL_BUDDY_OLLAMA_MODEL")
    or os.environ.get("SOVEREIGNJOURNAL_OLLAMA_MODEL")
    or "llama3:8b"
)
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_HISTORY_TURNS = 12
MAX_HISTORY_CHARS = 6000


SYSTEM_PROMPT = (
    "You are a warm, gentle companion for someone journaling privately on their own device. "
    "Their words matter; respond like a caring therapist who listens first—never preachy or clinical. "
    "Use short paragraphs, plain language, and a calm, friendly tone. "
    "Reflect back what you heard, name feelings lightly, and offer one small, optional thought or question—never a lecture. "
    "You may receive excerpts from their journal and the recent chat with them—use both; remember what they already told you in this conversation. "
    "If journal excerpts are included, weave them in naturally; don't quote long blocks. "
    "Never imply data leaves their device or goes to a cloud."
)


def _print_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")


def _run_librarian_query(question: str, top_k: int) -> dict[str, Any]:
    """
    Calls `librarian.py --query ... --json` so retrieval stays in Module B.
    Returns parsed JSON: { results: [...], errors: [...] }
    """
    librarian = Path(__file__).with_name("librarian.py")
    cmd = [
        sys.executable,
        str(librarian),
        "--query",
        question,
        "--top-k",
        str(int(top_k)),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"results": [], "errors": [proc.stderr.strip() or "librarian query failed"]}
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {"results": [], "errors": [f"Invalid JSON from librarian: {proc.stdout[:400]}"]}


def _normalize_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw[-MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    total = 0
    trimmed: list[dict[str, str]] = []
    for turn in reversed(history):
        total += len(turn["content"])
        if total > MAX_HISTORY_CHARS:
            break
        trimmed.append(turn)
    trimmed.reverse()
    return trimmed


def _build_journal_context_block(retrieved: list[dict[str, Any]]) -> tuple[str, list[str]]:
    context_lines: list[str] = []
    context_dates: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        meta = r.get("metadata") if isinstance(r, dict) else None
        date = ""
        source = ""
        if isinstance(meta, dict):
            date = str(meta.get("date", "")) if meta.get("date") is not None else ""
            source = str(meta.get("source_path", "")) if meta.get("source_path") is not None else ""
        if date:
            context_dates.append(date)
        text = str(r.get("text", "")).strip()
        if not text:
            continue
        context_lines.append(f"[{i}] date={date} source={source}\n{text}")

    dedup_dates: list[str] = []
    for d in context_dates:
        if d and d not in dedup_dates:
            dedup_dates.append(d)

    context_block = "\n\n".join(context_lines).strip()
    return context_block, dedup_dates


def _build_user_message(*, question: str, context_block: str) -> str:
    return (
        "They asked you something from the heart. Reply in a soft, human voice—like a supportive friend who gets it, "
        "not a textbook. Avoid bullet points unless they really help. "
        "Don't start with a disclaimer; don't list everything they should do. "
        "Use the journal excerpts below when they help; also stay consistent with what you already said earlier in this chat.\n\n"
        f"What they're wondering about now:\n{question}\n\n"
        f"Relevant journal excerpts (may be partial):\n{context_block if context_block else '(none yet — invite them to write or import entries)'}\n\n"
        "Your reply:"
    )


def _ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float = 120.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.62, "top_p": 0.92},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:400]}") from e
    except URLError as e:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {e}") from e

    try:
        j = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from Ollama: {body[:400]}") from e
    msg = j.get("message")
    if isinstance(msg, dict):
        return str(msg.get("content", "")).strip()
    return str(j.get("response", "")).strip()


def _ollama_generate(*, base_url: str, model: str, system: str, prompt: str, timeout_s: float = 120.0) -> str:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.62, "top_p": 0.92},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:400]}") from e
    except URLError as e:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {e}") from e

    try:
        j = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from Ollama: {body[:400]}") from e
    return str(j.get("response", "")).strip()


def answer(
    question: str,
    top_k: int,
    *,
    model: str,
    ollama_url: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    retrieved_json = _run_librarian_query(question, top_k)
    errors.extend([str(e) for e in retrieved_json.get("errors", []) if e])
    retrieved = retrieved_json.get("results", [])
    if not isinstance(retrieved, list):
        retrieved = []

    context_block, context_dates = _build_journal_context_block(retrieved)
    chat_history = _trim_history(history or [])
    user_message = _build_user_message(question=question, context_block=context_block)
    fallback_used = False

    try:
        if chat_history:
            messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(chat_history)
            messages.append({"role": "user", "content": user_message})
            resp = _ollama_chat(base_url=ollama_url, model=model, messages=messages)
        else:
            resp = _ollama_generate(
                base_url=ollama_url,
                model=model,
                system=SYSTEM_PROMPT,
                prompt=user_message,
            )
        if not resp:
            fallback_used = True
            resp = "I couldn't generate a response from your local model right now. Please try again."
    except Exception as e:
        fallback_used = True
        errors.append(str(e))
        resp = (
            "I couldn't reach your local Ollama model right now. "
            "Make sure Ollama is running and the model is ready, then try again."
        )

    return {
        "question": question,
        "answer": resp,
        "chunks_used": int(len(retrieved)),
        "fallback_used": bool(fallback_used),
        "context_dates": context_dates,
        "errors": errors,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="brain.py")
    p.add_argument("--ask", type=str, required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    p.add_argument(
        "--history-json",
        type=str,
        default="[]",
        help="JSON array of {role, content} prior chat turns",
    )
    args = p.parse_args(argv)

    try:
        history_raw = json.loads(args.history_json) if args.history_json.strip() else []
    except Exception:
        history_raw = []
    history = _normalize_history(history_raw)

    resp = answer(
        args.ask,
        args.top_k,
        model=args.model,
        ollama_url=args.ollama_url,
        history=history,
    )
    if args.json:
        _print_json(resp)
    else:
        sys.stdout.write(resp["answer"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
