from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    Split text into ~chunk_size character chunks with a small overlap for retrieval quality.
    """
    s = text.strip()
    if not s:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)

    out: list[Chunk] = []
    i = 0
    start = 0
    n = len(s)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = s[start:end].strip()
        if chunk:
            out.append(Chunk(index=i, text=chunk))
            i += 1
        if end >= n:
            break
        start = max(0, end - overlap)
    return out

