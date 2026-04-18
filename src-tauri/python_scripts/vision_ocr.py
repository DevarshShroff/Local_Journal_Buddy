from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OcrLine:
    text: str
    confidence: float
    # Normalized bounding box in Vision coords (x, y, w, h) in [0,1]
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class OcrResult:
    text: str
    lines: list[OcrLine]


class VisionOcrError(RuntimeError):
    pass


def _import_pyobjc() -> dict[str, Any]:
    """
    Import pyobjc frameworks lazily so this module can be imported on non-macOS.
    """
    try:
        import Quartz  # type: ignore
        import Vision  # type: ignore
        from Foundation import NSURL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise VisionOcrError(
            "pyobjc Vision frameworks not available. Install:\n"
            "  pip install pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-core pyobjc\n"
            f"Import error: {e}"
        ) from e

    return {"Quartz": Quartz, "Vision": Vision, "NSURL": NSURL}


def recognize_text(
    image_path: str,
    *,
    languages: list[str] | None = None,
    accurate: bool = True,
    use_language_correction: bool = True,
) -> OcrResult:
    """
    OCR using Apple's Vision framework (VNRecognizeTextRequest).
    Works well for printed + handwriting (best-effort; depends on OS model).

    Returns:
      - text: joined lines
      - lines: per-observation top candidate + confidence + bbox (normalized)
    """
    fw = _import_pyobjc()
    Quartz = fw["Quartz"]
    Vision = fw["Vision"]
    NSURL = fw["NSURL"]

    url = NSURL.fileURLWithPath_(image_path)
    ci_image = Quartz.CIImage.imageWithContentsOfURL_(url)
    if ci_image is None:
        raise VisionOcrError(f"Could not load image: {image_path}")

    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)

    # Use a Python list as a mutable capture for callback results
    captured: dict[str, Any] = {"error": None, "observations": None}

    def completion_handler(request, error) -> None:  # noqa: ANN001
        captured["error"] = error
        captured["observations"] = request.results()

    req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(completion_handler)

    if accurate:
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    else:
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)

    req.setUsesLanguageCorrection_(bool(use_language_correction))
    if languages:
        req.setRecognitionLanguages_(languages)
    else:
        req.setRecognitionLanguages_(["en-US"])

    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise VisionOcrError(str(err) if err is not None else "Vision performRequests failed")

    if captured["error"] is not None:
        raise VisionOcrError(str(captured["error"]))

    observations = captured["observations"] or []
    lines: list[OcrLine] = []
    for obs in observations:
        # Top candidate
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        cand = candidates[0]
        txt = str(cand.string())
        conf = float(cand.confidence())
        bbox = obs.boundingBox()
        bbox_t = (float(bbox.origin.x), float(bbox.origin.y), float(bbox.size.width), float(bbox.size.height))
        lines.append(OcrLine(text=txt, confidence=conf, bbox=bbox_t))

    # Keep reading order roughly top-to-bottom by bbox y (Vision origin is bottom-left).
    # Sort by y descending, then x ascending.
    def _sort_key(l: OcrLine) -> tuple[float, float]:
        if l.bbox is None:
            return (0.0, 0.0)
        x, y, _, _ = l.bbox
        return (-y, x)

    lines.sort(key=_sort_key)
    full_text = "\n".join(l.text for l in lines).strip()
    return OcrResult(text=full_text, lines=lines)

