"""
ocr_engine.py — Module A: The OCR Engine
Local_Journal_Buddy · macOS Vision Framework via pyobjc

Supports : .jpg, .jpeg, .png, .heic
Mode     : batch (process every supported image in a folder)
Output   : timestamped .txt file per image, written alongside the source image
           OR to a custom output directory.

Public API (consumed by downstream modules):
    ocr_image(image_path: Path) -> OCRResult
    ocr_folder(folder_path: Path, output_dir: Path | None) -> list[OCRResult]

Usage (CLI):
    python ocr_engine.py --folder ~/Desktop/journals
    python ocr_engine.py --folder ~/Desktop/journals --output ~/Desktop/txt_out
    python ocr_engine.py --image ~/Desktop/page1.heic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── macOS Vision via pyobjc ───────────────────────────────────────────────────
# Note: CoreImage is NOT imported — Vision loads images directly via NSURL.
try:
    import objc  # noqa: F401 — needed to initialise pyobjc runtime
    from Cocoa import NSURL
    from Vision import (
        VNImageRequestHandler,
        VNRecognizeTextRequest,
    )
except ImportError as exc:
    sys.exit(
        f"[ocr_engine] pyobjc not found: {exc}\n"
        "Install with: pip install pyobjc-framework-Vision pyobjc-framework-Cocoa pyobjc-framework-Quartz"
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ocr_engine")

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".heic"}
)

# Vision recognition level:
#   VNRequestTextRecognitionLevelAccurate  = 0  (slower, better for handwriting)
#   VNRequestTextRecognitionLevelFast      = 1
RECOGNITION_LEVEL_ACCURATE = 0


# ── Data contract (consumed by Module B / C) ──────────────────────────────────
@dataclass
class OCRResult:
    """Structured output handed off to the Librarian (Module B)."""

    source_path: str          # absolute path to the original image
    text: str                 # full extracted text (newline-separated blocks)
    timestamp: str            # ISO-8601 when OCR was performed
    word_count: int           # quick quality signal
    confidence_avg: float     # average Vision confidence score (0–1)
    txt_path: Optional[str]   # absolute path of saved .txt file (if written)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── Core OCR logic ─────────────────────────────────────────────────────────────
def _run_vision_ocr(image_path: Path) -> tuple[str, float]:
    """
    Call Apple Vision on *image_path*.
    Returns (extracted_text, average_confidence).
    Raises RuntimeError if Vision cannot process the file.
    """
    url = NSURL.fileURLWithPath_(str(image_path.resolve()))

    # Build the recognition request
    request = VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(RECOGNITION_LEVEL_ACCURATE)
    request.setUsesLanguageCorrection_(True)  # helps with cursive / joined letters

    # Create a handler and perform the request
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    success, error = handler.performRequests_error_([request], None)

    if not success or error:
        raise RuntimeError(f"Vision framework error: {error}")

    observations = request.results()
    if not observations:
        return "", 0.0

    lines: list[str] = []
    confidences: list[float] = []

    for obs in observations:
        # Top candidate for each text block
        candidate = obs.topCandidates_(1)
        if candidate and len(candidate) > 0:
            top = candidate[0]
            lines.append(top.string())
            confidences.append(top.confidence())

    text = "\n".join(lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf


def _write_txt(text: str, source_path: Path, output_dir: Optional[Path]) -> Path:
    """Persist extracted text as a .txt file; returns its path."""
    stem = source_path.stem
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}_{timestamp_str}.txt"

    dest_dir = output_dir if output_dir else source_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    txt_path = dest_dir / filename
    txt_path.write_text(text, encoding="utf-8")
    return txt_path


# ── Public API ─────────────────────────────────────────────────────────────────
def ocr_image(
    image_path: Path | str,
    output_dir: Optional[Path | str] = None,
    save_txt: bool = True,
) -> OCRResult:
    """
    OCR a single image.

    Parameters
    ----------
    image_path : path to .jpg/.png/.heic file
    output_dir : where to write the .txt; defaults to same dir as image
    save_txt   : set False to skip writing to disk (useful when Module B
                 stores text directly in SQLite/ChromaDB)

    Returns
    -------
    OCRResult  — pass directly to Module B's ingest pipeline
    """
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    timestamp = datetime.now().isoformat()
    errors: list[str] = []
    text = ""
    avg_conf = 0.0
    txt_path_str: Optional[str] = None

    if not image_path.exists():
        errors.append(f"File not found: {image_path}")
        logger.error(errors[-1])
    elif image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        errors.append(
            f"Unsupported format '{image_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        logger.error(errors[-1])
    else:
        try:
            logger.info("Processing: %s", image_path.name)
            text, avg_conf = _run_vision_ocr(image_path)

            if save_txt and text.strip():
                txt_path = _write_txt(text, image_path, output_dir)
                txt_path_str = str(txt_path)
                logger.info("  ✓ saved → %s", txt_path.name)
            elif not text.strip():
                logger.warning("  ⚠ no text detected in %s", image_path.name)

        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            logger.error("  ✗ %s", exc)

    return OCRResult(
        source_path=str(image_path),
        text=text,
        timestamp=timestamp,
        word_count=len(text.split()) if text else 0,
        confidence_avg=round(avg_conf, 4),
        txt_path=txt_path_str,
        errors=errors,
    )


def ocr_folder(
    folder_path: Path | str,
    output_dir: Optional[Path | str] = None,
    save_txt: bool = True,
) -> list[OCRResult]:
    """
    Batch-OCR every supported image in *folder_path* (non-recursive).

    Parameters
    ----------
    folder_path : directory containing journal photos
    output_dir  : shared output dir for all .txt files; defaults to each
                  image's own directory
    save_txt    : passed through to ocr_image()

    Returns
    -------
    list[OCRResult] — ordered by filename; empty list if no images found
    """
    folder_path = Path(folder_path).expanduser().resolve()

    if not folder_path.is_dir():
        logger.error("Not a directory: %s", folder_path)
        return []

    images = sorted(
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        logger.warning("No supported images found in %s", folder_path)
        return []

    logger.info("Found %d image(s) in '%s'", len(images), folder_path.name)

    results: list[OCRResult] = []
    for img in images:
        result = ocr_image(img, output_dir=output_dir, save_txt=save_txt)
        results.append(result)

    # Summary
    success = sum(1 for r in results if not r.errors and r.text.strip())
    logger.info(
        "Batch complete — %d/%d succeeded · avg confidence %.2f",
        success,
        len(results),
        sum(r.confidence_avg for r in results) / len(results) if results else 0,
    )
    return results


# ── CLI entry point ────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr_engine",
        description="Local_Journal_Buddy — Module A: OCR Engine (macOS Vision)",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="OCR a single image file")
    group.add_argument("--folder", metavar="PATH", help="Batch-OCR a folder of images")
    p.add_argument(
        "--output", metavar="DIR",
        help="Directory for output .txt files (default: same as input)"
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Print text to stdout only; do not write .txt files"
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print OCRResult(s) as JSON to stdout"
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    save_txt = not args.no_save

    if args.image:
        result = ocr_image(args.image, output_dir=args.output, save_txt=save_txt)
        if args.json:
            print(result.to_json())
        else:
            print(result.text)

    else:  # --folder
        results = ocr_folder(args.folder, output_dir=args.output, save_txt=save_txt)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(f"\n── {Path(r.source_path).name} ──")
                print(r.text or "[no text detected]")


if __name__ == "__main__":
    main()