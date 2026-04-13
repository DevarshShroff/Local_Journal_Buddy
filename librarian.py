"""
librarian.py — Module B: The Librarian (Vector Database)
Local_Journal_Buddy · ChromaDB + Sentence-Transformers · 100% offline

Responsibilities:
  1. Chunking  — split journal text into 500-char overlapping segments
  2. Embedding — encode chunks via local sentence-transformers model
  3. Persistence— store in ~/Library/Application Support/SovereignJournal/chroma/
  4. Retrieval  — semantic search across ALL entries, return top-k chunks

Public API (consumed by Module C / Tauri backend):
    ingest_result(ocr_result)          -> IngestReport
    ingest_text(text, date, source)    -> IngestReport
    query(question, top_k)             -> list[JournalChunk]
    delete_entry(source_path)          -> int   (chunks removed)
    get_all_entries()                  -> list[dict]

Usage (CLI):
    python librarian.py --ingest-file output/page_001.txt --date 2024-03-15
    python librarian.py --query "when did I feel overwhelmed"
    python librarian.py --query "moments I felt proud" --top-k 5
    python librarian.py --list-entries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("librarian")

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_DIR = Path.home() / "Library" / "Application Support" / "SovereignJournal" / "chroma"
COLLECTION_NAME = "journal_entries"

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 75    # characters — keeps context across chunk boundaries

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~22 MB, fast on CPU, great for semantic similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Lazy imports ──────────────────────────────────────────────────────────────
def _require(package: str, install: str):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        sys.exit(
            f"[librarian] Missing: '{package}'\n"
            f"Install: pip install {install}"
        )


# ── Data contracts ────────────────────────────────────────────────────────────
@dataclass
class JournalChunk:
    """A single chunk returned from a semantic query — fed to Module C."""
    chunk_id:     str
    text:         str
    date:         str
    source_path:  str
    chunk_index:  int
    total_chunks: int
    distance:     float          # lower = more similar (0.0 is identical)
    word_count:   int
    corrected:    bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestReport:
    """Summary returned after ingesting an OCR result or text block."""
    source_path:   str
    date:          str
    chunks_stored: int
    chunk_ids:     list[str]
    skipped:       int           # chunks already in DB (dedup)
    errors:        list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── Singleton: embedding model ────────────────────────────────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    st = _require("sentence_transformers", "sentence-transformers")
    logger.info("Loading embedding model '%s' (first run downloads ~22 MB)…", EMBEDDING_MODEL)
    _embedder = st.SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model ready.")
    return _embedder


# ── Singleton: ChromaDB collection ────────────────────────────────────────────
_collection = None

def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    chromadb = _require("chromadb", "chromadb")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Use cosine similarity — better for semantic text search than L2
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' ready at %s", COLLECTION_NAME, DB_DIR)
    return _collection


# ── Chunking ──────────────────────────────────────────────────────────────────
def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping 500-char segments.
    Tries to break on sentence boundaries ('. ') to keep meaning intact.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            # Try to break at the last sentence boundary within the window
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + (CHUNK_SIZE // 2):
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap so context isn't lost at boundaries
        start = end - CHUNK_OVERLAP

    return chunks


def _make_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    """
    Deterministic ID — same image + same chunk always produces the same ID.
    Enables safe re-ingestion (dedup) without scanning the whole DB.
    """
    fingerprint = f"{source_path}::{chunk_index}::{text[:50]}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


# ── Core ingest logic ─────────────────────────────────────────────────────────
def _embed(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    vectors = embedder.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def _ingest(
    text: str,
    date: str,
    source_path: str,
    corrected: bool = False,
    confidence: float = 0.0,
) -> IngestReport:
    collection = _get_collection()
    errors: list[str] = []
    stored_ids: list[str] = []
    skipped = 0

    chunks = _chunk_text(text)
    if not chunks:
        return IngestReport(
            source_path=source_path,
            date=date,
            chunks_stored=0,
            chunk_ids=[],
            skipped=0,
            errors=["No text to ingest after chunking."],
        )

    total = len(chunks)
    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(source_path, i, chunk)

        # Dedup — skip if this exact chunk is already stored
        existing = collection.get(ids=[chunk_id])
        if existing["ids"]:
            skipped += 1
            continue

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "date":         date,
            "source_path":  source_path,
            "chunk_index":  i,
            "total_chunks": total,
            "word_count":   len(chunk.split()),
            "corrected":    str(corrected),   # ChromaDB metadata must be str/int/float
            "confidence":   round(confidence, 4),
            "ingested_at":  datetime.now().isoformat(),
        })

    if ids:
        try:
            embeddings = _embed(documents)
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            stored_ids = ids
            logger.info(
                "Ingested '%s' → %d chunk(s) stored, %d skipped.",
                Path(source_path).name, len(ids), skipped,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            logger.error("Ingest failed: %s", exc)
    else:
        logger.info("All chunks already in DB — skipped '%s'.", Path(source_path).name)

    return IngestReport(
        source_path=source_path,
        date=date,
        chunks_stored=len(stored_ids),
        chunk_ids=stored_ids,
        skipped=skipped,
        errors=errors,
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def ingest_result(ocr_result, corrected: bool = False) -> IngestReport:
    """
    Ingest directly from an OCRResult (Module A output).

    Usage:
        from ocr_engine import ocr_image
        from ocr_corrector import correct_ocr_result
        from librarian import ingest_result

        raw   = ocr_image("journals/page.heic", save_txt=False)
        clean = correct_ocr_result(raw)
        report = ingest_result(clean, corrected=True)
    """
    date = ocr_result.timestamp[:10]   # ISO date portion: "2024-03-15"
    return _ingest(
        text=ocr_result.text,
        date=date,
        source_path=ocr_result.source_path,
        corrected=corrected,
        confidence=ocr_result.confidence_avg,
    )


def ingest_text(
    text: str,
    date: Optional[str] = None,
    source_path: str = "manual_entry",
    corrected: bool = False,
) -> IngestReport:
    """
    Ingest a raw string directly (useful for testing or typed entries).
    date defaults to today if not provided.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    return _ingest(text=text, date=date, source_path=source_path, corrected=corrected)


def query(question: str, top_k: int = 3) -> list[JournalChunk]:
    """
    Semantic search across ALL journal entries.
    Returns top_k most relevant chunks — passed directly to Module C (Ollama).

    Usage:
        chunks = query("when did I feel overwhelmed at work")
        for c in chunks:
            print(c.date, c.text)
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("Database is empty — ingest some journal entries first.")
        return []

    question_vec = _embed([question])[0]

    results = collection.query(
        query_embeddings=[question_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[JournalChunk] = []
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        chunks.append(JournalChunk(
            chunk_id=chunk_id,
            text=results["documents"][0][i],
            date=meta.get("date", "unknown"),
            source_path=meta.get("source_path", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            total_chunks=int(meta.get("total_chunks", 1)),
            distance=round(results["distances"][0][i], 4),
            word_count=int(meta.get("word_count", 0)),
            corrected=meta.get("corrected", "False") == "True",
        ))

    logger.info("Query returned %d chunk(s) for: '%s'", len(chunks), question)
    return chunks


def delete_entry(source_path: str) -> int:
    """
    Remove all chunks belonging to a given source image.
    Returns number of chunks deleted.
    """
    collection = _get_collection()
    results = collection.get(where={"source_path": source_path})
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
        logger.info("Deleted %d chunk(s) for '%s'.", len(ids), source_path)
    return len(ids)


def get_all_entries() -> list[dict]:
    """
    Return a deduplicated list of all ingested journal entries (one per source image).
    Useful for the Dashboard calendar view in Module D.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in results["metadatas"]:
        src = meta.get("source_path", "")
        if src not in seen:
            seen[src] = {
                "source_path": src,
                "date":        meta.get("date", ""),
                "ingested_at": meta.get("ingested_at", ""),
                "total_chunks": meta.get("total_chunks", 1),
            }
    return sorted(seen.values(), key=lambda x: x["date"], reverse=True)


# ── CLI ────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="librarian",
        description="Local_Journal_Buddy — Module B: The Librarian",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ingest-file", metavar="PATH", help="Ingest a .txt file into the DB")
    g.add_argument("--ingest-text", metavar="STRING", help="Ingest a string directly")
    g.add_argument("--query",       metavar="QUESTION", help="Semantic search")
    g.add_argument("--list-entries", action="store_true", help="List all ingested entries")
    g.add_argument("--delete",      metavar="SOURCE_PATH", help="Delete all chunks for a source")

    p.add_argument("--date",  metavar="YYYY-MM-DD", help="Date for ingestion (default: today)")
    p.add_argument("--top-k", metavar="N", type=int, default=3, help="Results to return (default: 3)")
    p.add_argument("--json",  action="store_true", help="Output as JSON")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.ingest_file:
        text = Path(args.ingest_file).expanduser().read_text(encoding="utf-8")
        report = ingest_text(text, date=args.date, source_path=args.ingest_file)
        print(report.to_json() if args.json else
              f"Stored {report.chunks_stored} chunk(s), skipped {report.skipped}.")

    elif args.ingest_text:
        report = ingest_text(args.ingest_text, date=args.date)
        print(report.to_json() if args.json else
              f"Stored {report.chunks_stored} chunk(s), skipped {report.skipped}.")

    elif args.query:
        chunks = query(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps([c.to_dict() for c in chunks], indent=2))
        else:
            for c in chunks:
                print(f"\n── {c.date}  (similarity: {1 - c.distance:.0%}) ──")
                print(c.text)

    elif args.list_entries:
        entries = get_all_entries()
        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            for e in entries:
                print(f"{e['date']}  {Path(e['source_path']).name}  ({e['total_chunks']} chunks)")

    elif args.delete:
        n = delete_entry(args.delete)
        print(f"Deleted {n} chunk(s).")


if __name__ == "__main__":
    main()