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

DEFAULT_MODEL = os.environ.get("SOVEREIGNJOURNAL_OLLAMA_MODEL", "llama3:8b")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


SYSTEM_PROMPT = (
    "You are a warm, gentle companion for someone journaling privately on their own device. "
    "Their words matter; respond like a caring therapist who listens first—never preachy or clinical. "
    "Use short paragraphs, plain language, and a calm, friendly tone. "
    "Reflect back what you heard, name feelings lightly, and offer one small, optional thought or question—never a lecture. "
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


def _build_prompt(*, question: str, retrieved: list[dict[str, Any]]) -> tuple[str, list[str]]:
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

    # De-dup dates but preserve order
    dedup_dates: list[str] = []
    for d in context_dates:
        if d and d not in dedup_dates:
            dedup_dates.append(d)

    context_block = "\n\n".join(context_lines).strip()
    user_prompt = (
        "They asked you something from the heart. Reply in a soft, human voice—like a supportive friend who gets it, "
        "not a textbook. Avoid bullet points unless they really help. "
        "Don't start with a disclaimer; don't list everything they should do. "
        "If there's little or no journal context below, say so kindly and invite them to share more when they're ready.\n\n"
        f"What they're wondering about:\n{question}\n\n"
        f"Relevant journal bits (may be partial):\n{context_block if context_block else '(none yet)'}\n\n"
        "Your reply:"
    )
    return user_prompt, dedup_dates


def _ollama_generate(*, base_url: str, model: str, system: str, prompt: str, timeout_s: float = 90.0) -> str:
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


def answer(question: str, top_k: int, *, model: str, ollama_url: str) -> dict[str, Any]:
    errors: list[str] = []
    retrieved_json = _run_librarian_query(question, top_k)
    errors.extend([str(e) for e in retrieved_json.get("errors", []) if e])
    retrieved = retrieved_json.get("results", [])
    if not isinstance(retrieved, list):
        retrieved = []

    prompt, context_dates = _build_prompt(question=question, retrieved=retrieved)
    fallback_used = False

    try:
        resp = _ollama_generate(
            base_url=ollama_url,
            model=model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            timeout_s=90.0,
        )
        if not resp:
            fallback_used = True
            resp = "I couldn’t generate a response from your local model right now. Please try again."
    except Exception as e:
        fallback_used = True
        errors.append(str(e))
        resp = (
            "I couldn’t reach your local Ollama model right now. "
            "Make sure Ollama is running (ollama serve) and the model is pulled, then try again."
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
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--json", action="store_true")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    args = p.parse_args(argv)

    resp = answer(args.ask, args.top_k, model=args.model, ollama_url=args.ollama_url)
    if args.json:
        _print_json(resp)
    else:
        sys.stdout.write(resp["answer"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

