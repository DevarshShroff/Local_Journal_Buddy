"""
test_pipeline.py — Full Pipeline Test
Local_Journal_Buddy · A → B → C end-to-end smoke test

Usage:
    python test_pipeline.py --image ~/Desktop/notes-1.jpg
    python test_pipeline.py --text "Today was really hard. I felt overwhelmed at work and couldn't focus."
    python test_pipeline.py --text "..." --skip-ocr --skip-correction
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────
def header(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def ok(msg: str):    print(f"  ✓  {msg}")
def warn(msg: str):  print(f"  ⚠  {msg}")
def fail(msg: str):  print(f"  ✗  {msg}"); sys.exit(1)
def info(msg: str):  print(f"     {msg}")

def preview(text: str, chars: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    return textwrap.shorten(text, width=chars, placeholder="…")


# ── Step 1: OCR ───────────────────────────────────────────────────────────────
def test_ocr(image_path: str) -> str:
    header("STEP 1 — OCR Engine (Module A)")
    try:
        from ocr_engine import ocr_image
        result = ocr_image(image_path, save_txt=False)

        if result.errors:
            fail(f"OCR errors: {result.errors}")

        if not result.text.strip():
            fail("OCR returned empty text. Check image quality or path.")

        ok(f"Text extracted — {result.word_count} words, "
           f"confidence {result.confidence_avg:.0%}")
        info(f"Preview: \"{preview(result.text)}\"")
        return result.text

    except ImportError:
        fail("ocr_engine.py not found. Make sure it's in the same folder.")


# ── Step 2: Correction ────────────────────────────────────────────────────────
def test_correction(text: str) -> str:
    header("STEP 2 — OCR Corrector (Module A.1)")
    try:
        from ocr_corrector import correct
        result = correct(text)

        if result.errors:
            warn(f"Corrector warnings: {result.errors}")
        else:
            ok(f"Corrections applied — "
               f"{result.spell_corrections} spell, "
               f"{result.grammar_corrections} grammar")

        if result.total_corrections > 0:
            info(f"Before: \"{preview(result.original_text)}\"")
            info(f"After:  \"{preview(result.corrected_text)}\"")
        else:
            info("Text was already clean — no changes needed.")

        return result.corrected_text

    except ImportError:
        fail("ocr_corrector.py not found.")


# ── Step 3: Ingest into ChromaDB ──────────────────────────────────────────────
def test_ingest(text: str, source: str) -> bool:
    header("STEP 3 — Librarian Ingest (Module B)")
    try:
        from librarian import ingest_text, get_all_entries
        report = ingest_text(
            text=text,
            date="2024-01-01",        # fixed date for repeatable test
            source_path=source,
            corrected=True,
        )

        if report.errors:
            warn(f"Ingest warnings: {report.errors}")

        ok(f"Stored {report.chunks_stored} chunk(s), "
           f"skipped {report.skipped} duplicate(s)")
        info(f"DB now contains {len(get_all_entries())} unique entry/entries")
        return True

    except ImportError:
        fail("librarian.py not found.")


# ── Step 4: Semantic query ────────────────────────────────────────────────────
def test_query() -> bool:
    header("STEP 4 — Librarian Query (Module B)")
    try:
        from librarian import query
        TEST_QUERY = "feeling overwhelmed or stressed"
        chunks = query(TEST_QUERY, top_k=2)

        if not chunks:
            warn("Query returned no results. "
                 "DB may be empty — run ingest first.")
            return False

        ok(f"Retrieved {len(chunks)} chunk(s) for: \"{TEST_QUERY}\"")
        for i, c in enumerate(chunks, 1):
            info(f"  [{i}] {c.date} · similarity {1 - c.distance:.0%} · "
                 f"\"{preview(c.text, 120)}\"")
        return True

    except ImportError:
        fail("librarian.py not found.")


# ── Step 5: Brain / Ollama ────────────────────────────────────────────────────
def test_brain() -> bool:
    header("STEP 5 — The Brain (Module C)")
    try:
        from brain import health_check, ask
        status = health_check()

        if not status.reachable:
            warn("Ollama is NOT running — testing fallback mode instead.")
            info("To start Ollama:  ollama serve")
            info(f"To pull model:    ollama pull {status.model}")
        elif not status.model_available:
            warn(f"Ollama is running but model '{status.model}' is missing.")
            info(f"Fix:  ollama pull {status.model}")
        else:
            ok(f"Ollama reachable · model '{status.model}' available")

        # Ask a test question regardless (fallback will kick in if needed)
        TEST_QUESTION = "How have I been feeling lately?"
        info(f"Asking: \"{TEST_QUESTION}\"")
        response = ask(TEST_QUESTION, top_k=2)

        if response.fallback_used:
            warn("Fallback mode active — raw journal chunks shown instead of LLM response")
        else:
            ok("LLM response received")

        info(f"Answer preview: \"{preview(response.answer, 300)}\"")
        info(f"Chunks used: {response.chunks_used} · "
             f"Dates: {', '.join(response.context_dates) or 'none'}")
        return True

    except ImportError:
        fail("brain.py not found.")


# ── Summary ───────────────────────────────────────────────────────────────────
def summary(results: dict):
    header("TEST SUMMARY")
    all_passed = True
    for step, passed in results.items():
        icon = "✓" if passed else "⚠"
        print(f"  {icon}  {step}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🟢  All steps passed — pipeline is healthy!")
    else:
        print("  🟡  Some steps had warnings — check output above.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="test_pipeline",
        description="Local_Journal_Buddy — end-to-end pipeline smoke test",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", metavar="PATH",   help="Test with a real journal image")
    src.add_argument("--text",  metavar="STRING", help="Test with a hardcoded string (skips OCR)")

    parser.add_argument("--skip-correction", action="store_true",
                        help="Skip Module A.1 (ocr_corrector)")
    args = parser.parse_args()

    results = {}

    # ── Source text
    if args.image:
        raw_text = test_ocr(args.image)
        results["Module A  · OCR Engine"] = bool(raw_text)
        source_label = args.image
    else:
        header("STEP 1 — OCR Engine (Module A)")
        raw_text = args.text
        info("Using provided text — skipping OCR.")
        results["Module A  · OCR Engine"] = None   # skipped
        source_label = "test_pipeline_manual"

    # ── Correction
    if not args.skip_correction:
        corrected_text = test_correction(raw_text)
        results["Module A.1· OCR Corrector"] = bool(corrected_text)
    else:
        corrected_text = raw_text
        results["Module A.1· OCR Corrector"] = None  # skipped

    # ── Ingest
    ingest_ok = test_ingest(corrected_text, source=source_label)
    results["Module B  · Librarian Ingest"] = ingest_ok

    # ── Query
    query_ok = test_query()
    results["Module B  · Librarian Query"] = query_ok

    # ── Brain
    brain_ok = test_brain()
    results["Module C  · Brain (Ollama)"] = brain_ok

    # ── Summary
    # Filter out skipped steps
    filtered = {k: v for k, v in results.items() if v is not None}
    summary(filtered)


if __name__ == "__main__":
    main()
    