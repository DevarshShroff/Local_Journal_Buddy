from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from paths import app_support_dir, ensure_dir
from vision_ocr import VisionOcrError, recognize_text


def _print_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_output_dir() -> Path:
    # Architecture spec: ~/Library/Application Support/SovereignJournal/
    return ensure_dir(app_support_dir("SovereignJournal") / "ocr")


def _save_txt(text: str, *, image_path: str, out_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(image_path).stem or "entry"
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:80] or "entry"
    p = out_dir / f"{ts}_{safe_stem}.txt"
    p.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return p


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ocr_engine.py")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out-dir", type=str, default="", help="Override OCR output directory")
    p.add_argument("--lang", action="append", default=[], help="Recognition language (repeatable). Default: en-US")
    p.add_argument("--fast", action="store_true", help="Use fast recognition (less accurate)")
    args = p.parse_args(argv)

    image_path = args.image
    timestamp = utc_now_iso()

    # Allow integration tests / sandboxed contexts to redirect storage.
    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        out_dir = _default_output_dir()

    errors: list[str] = []
    saved_txt_path: str | None = None
    lines_json: list[dict[str, object]] = []
    text = ""
    confidence_avg = 0.0

    try:
        langs = args.lang if args.lang else ["en-US"]
        res = recognize_text(
            image_path,
            languages=langs,
            accurate=not args.fast,
            use_language_correction=True,
        )
        text = res.text
        if res.lines:
            confidence_avg = sum(l.confidence for l in res.lines) / max(1, len(res.lines))
            for l in res.lines:
                lines_json.append(
                    {
                        "text": l.text,
                        "confidence": l.confidence,
                        "bbox": list(l.bbox) if l.bbox is not None else None,
                    }
                )

        if not args.no_save:
            saved = _save_txt(text, image_path=image_path, out_dir=out_dir)
            saved_txt_path = str(saved)

    except VisionOcrError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"OCR failed: {e}")

    word_count = len([w for w in text.split() if w.strip()])

    result = {
        "source_path": image_path,
        "timestamp": timestamp,
        "text": text,
        "word_count": word_count,
        "confidence_avg": float(confidence_avg),
        "saved_txt_path": saved_txt_path,
        "out_dir": str(out_dir),
        "lines": lines_json,
        "errors": errors,
    }

    if args.json:
        _print_json(result)
    else:
        if errors:
            sys.stderr.write(errors[0] + "\n")
            return 2
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

