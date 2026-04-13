"""
ocr_corrector.py — Module A.1: Local OCR Post-Correction
Local_Journal_Buddy · 100% offline, no cloud calls

Two-layer correction pipeline:
  Layer 1 · SymSpell   — fast dictionary spell correction
                         (catches: "teh"→"the", "lfe"→"life", "adn"→"and")
  Layer 2 · LanguageTool (local JVM) — grammar + context correction
                         (catches: punctuation, capitalisation, word boundaries)

Public API (consumed by ocr_engine.py or Module B):
    correct(text: str) -> CorrectionResult
    correct_result(ocr_result: OCRResult) -> OCRResult   # patches in-place copy

Install:
    pip install symspellpy language-tool-python
    (LanguageTool auto-downloads its Java server on first run, ~200 MB, once only)

Usage (CLI):
    echo "I fel so tierd todya" | python ocr_corrector.py
    python ocr_corrector.py --text "I fel so tierd todya"
    python ocr_corrector.py --file output/page_001_20240101.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ocr_corrector")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── Lazy imports (fail gracefully with install hint) ──────────────────────────
def _require(package: str, install: str) -> object:
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        sys.exit(f"[ocr_corrector] Missing: '{package}'\nInstall: pip install {install}")


# ── Constants ─────────────────────────────────────────────────────────────────
# SymSpell dictionary — bundled inside symspellpy package
_SYMSPELL_DICT_SUBPATH = "frequency_dictionary_en_82_765.txt"
_SYMSPELL_BIGRAM_SUBPATH = "frequency_bigramdictionary_en_243_342.txt"

# Max edit distance for spell correction
# 2 is the sweet spot: catches most OCR noise without over-correcting
MAX_EDIT_DISTANCE = 2

# Words to never correct (journal-specific, add your own)
PROTECTED_WORDS: frozenset[str] = frozenset({
    "i",        # personal pronoun — SymSpell sometimes maps to "a"
    "ok", "okay",
    "gonna", "wanna", "kinda", "sorta",   # casual writing
})


# ── Data contract ─────────────────────────────────────────────────────────────
@dataclass
class CorrectionResult:
    original_text: str
    corrected_text: str
    spell_corrections: int      # how many words SymSpell changed
    grammar_corrections: int    # how many issues LanguageTool fixed
    errors: list[str] = field(default_factory=list)

    @property
    def total_corrections(self) -> int:
        return self.spell_corrections + self.grammar_corrections

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── Layer 1 · SymSpell spell correction ───────────────────────────────────────
_symspell_instance = None

def _get_symspell():
    global _symspell_instance
    if _symspell_instance is not None:
        return _symspell_instance

    sym_mod = _require("symspellpy", "symspellpy")
    SymSpell = sym_mod.SymSpell
    Verbosity = sym_mod.Verbosity

    import pkg_resources
    dict_path   = pkg_resources.resource_filename("symspellpy", _SYMSPELL_DICT_SUBPATH)
    bigram_path = pkg_resources.resource_filename("symspellpy", _SYMSPELL_BIGRAM_SUBPATH)

    sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=7)
    sym.load_dictionary(dict_path, term_index=0, count_index=1)
    sym.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    _symspell_instance = (sym, Verbosity)
    logger.info("SymSpell dictionary loaded.")
    return _symspell_instance


def _spell_correct_line(line: str) -> tuple[str, int]:
    """
    Correct a single line using SymSpell compound correction.
    Returns (corrected_line, num_changes).
    Preserves blank lines and lines that are purely punctuation/numbers.
    """
    if not line.strip() or re.fullmatch(r"[\W\d\s]+", line):
        return line, 0

    sym, Verbosity = _get_symspell()

    # Protect special tokens before correction
    protected: dict[str, str] = {}
    working = line
    for i, word in enumerate(PROTECTED_WORDS):
        # Case-insensitive whole-word protection
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        placeholder = f"__PROT{i}__"
        if pattern.search(working):
            protected[placeholder] = pattern.findall(working)[0]
            working = pattern.sub(placeholder, working)

    suggestions = sym.lookup_compound(working, max_edit_distance=MAX_EDIT_DISTANCE)
    if not suggestions:
        return line, 0

    corrected = suggestions[0].term

    # Restore protected words
    for placeholder, original in protected.items():
        corrected = corrected.replace(placeholder.lower(), original)
        corrected = corrected.replace(placeholder, original)

    changes = sum(
        1 for a, b in zip(line.split(), corrected.split()) if a.lower() != b.lower()
    )
    return corrected, changes


def _apply_spell_correction(text: str) -> tuple[str, int]:
    """Apply SymSpell line-by-line (preserves paragraph structure)."""
    lines = text.splitlines()
    corrected_lines = []
    total_changes = 0
    for line in lines:
        c_line, n = _spell_correct_line(line)
        corrected_lines.append(c_line)
        total_changes += n
    return "\n".join(corrected_lines), total_changes


# ── Layer 2 · LanguageTool grammar correction ─────────────────────────────────
_lt_instance = None

def _get_language_tool():
    global _lt_instance
    if _lt_instance is not None:
        return _lt_instance

    lt_mod = _require("language_tool_python", "language-tool-python")

    logger.info("Starting local LanguageTool server (first run may take ~30 s)…")
    # language_tool_python.LanguageTool downloads the JVM server once to ~/.cache
    tool = lt_mod.LanguageTool("en-US")
    _lt_instance = tool
    logger.info("LanguageTool ready.")
    return _lt_instance


# Rules to skip — these fire too aggressively on informal journal writing
_LT_DISABLED_RULES: set[str] = {
    "UPPERCASE_SENTENCE_START",   # journal lines often don't start with capitals
    "COMMA_PARENTHESIS_WHITESPACE",
    "EN_QUOTES",
    "MORFOLOGIK_RULE_EN_US",      # SymSpell already handled spelling
}


def _apply_grammar_correction(text: str) -> tuple[str, int]:
    """Apply LanguageTool corrections, skipping rules that over-fire on journals."""
    tool = _get_language_tool()

    import language_tool_python
    matches = tool.check(text)

    # Filter out disabled rules
    matches = [m for m in matches if m.ruleId not in _LT_DISABLED_RULES]

    if not matches:
        return text, 0

    corrected = language_tool_python.utils.correct(text, matches)
    return corrected, len(matches)


# ── Public API ─────────────────────────────────────────────────────────────────
def correct(
    text: str,
    use_spell: bool = True,
    use_grammar: bool = True,
) -> CorrectionResult:
    """
    Run the two-layer correction pipeline on raw OCR text.

    Parameters
    ----------
    text        : raw string from ocr_engine OCRResult.text
    use_spell   : enable SymSpell (Layer 1)
    use_grammar : enable LanguageTool (Layer 2)

    Returns
    -------
    CorrectionResult — pass corrected_text to Module B for ingestion
    """
    errors: list[str] = []
    spell_n = 0
    grammar_n = 0
    working = text

    if use_spell:
        try:
            working, spell_n = _apply_spell_correction(working)
            logger.info("SymSpell: %d correction(s)", spell_n)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SymSpell error: {exc}")
            logger.error("SymSpell failed: %s", exc)

    if use_grammar:
        try:
            working, grammar_n = _apply_grammar_correction(working)
            logger.info("LanguageTool: %d correction(s)", grammar_n)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"LanguageTool error: {exc}")
            logger.error("LanguageTool failed: %s", exc)

    return CorrectionResult(
        original_text=text,
        corrected_text=working,
        spell_corrections=spell_n,
        grammar_corrections=grammar_n,
        errors=errors,
    )


def correct_ocr_result(ocr_result, **kwargs) -> object:
    """
    Convenience wrapper: takes an OCRResult from ocr_engine.py,
    runs correction, returns a NEW OCRResult with corrected text.
    (Does not mutate the original.)

    Usage:
        from ocr_engine import ocr_image
        from ocr_corrector import correct_ocr_result

        result = ocr_image("page.heic")
        clean  = correct_ocr_result(result)
        # clean.text is now corrected
    """
    import copy
    cr = correct(ocr_result.text, **kwargs)
    new_result = copy.copy(ocr_result)
    new_result.text = cr.corrected_text
    new_result.word_count = len(cr.corrected_text.split())
    # Attach correction stats as a note (Module B can store this)
    if not hasattr(new_result, "correction_stats"):
        object.__setattr__(new_result, "correction_stats", cr.to_dict())
    return new_result


# ── CLI ────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr_corrector",
        description="Local_Journal_Buddy — Module A.1: OCR Post-Correction",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", metavar="STRING", help="Correct a string directly")
    src.add_argument("--file", metavar="PATH",   help="Correct a .txt file")
    p.add_argument("--no-spell",   action="store_true", help="Skip SymSpell layer")
    p.add_argument("--no-grammar", action="store_true", help="Skip LanguageTool layer")
    p.add_argument("--json",       action="store_true", help="Output CorrectionResult as JSON")
    p.add_argument("--overwrite",  action="store_true", help="Overwrite input .txt file with corrected text")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Read input
    if args.text:
        raw = args.text
    elif args.file:
        raw = Path(args.file).expanduser().read_text(encoding="utf-8")
    else:
        # Read from stdin if no flag given
        raw = sys.stdin.read()

    if not raw.strip():
        print("[ocr_corrector] Empty input — nothing to correct.", file=sys.stderr)
        sys.exit(0)

    result = correct(
        raw,
        use_spell=not args.no_spell,
        use_grammar=not args.no_grammar,
    )

    if args.json:
        print(result.to_json())
    else:
        print(result.corrected_text)

    if args.file and args.overwrite:
        Path(args.file).expanduser().write_text(result.corrected_text, encoding="utf-8")
        logger.info("Overwrote %s with corrected text.", args.file)

    logger.info(
        "Done — %d spell + %d grammar = %d total correction(s).",
        result.spell_corrections,
        result.grammar_corrections,
        result.total_corrections,
    )


if __name__ == "__main__":
    main()