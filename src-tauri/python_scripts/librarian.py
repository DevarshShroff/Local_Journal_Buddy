from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from chunker import chunk_text
from embedding import EmbeddingError, default_embedder
from sovereign_store import (
    chroma_dir,
    connect,
    delete_entry_row_and_file,
    delete_entry_row_and_file_by_id,
    get_entry,
    get_entry_by_id,
    init_db,
    list_entries as sql_list_entries,
    save_entry_text,
    upsert_entry,
)


def _print_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")


def cmd_list_entries(as_json: bool) -> int:
    conn = connect()
    init_db(conn)
    entries = sql_list_entries(conn)

    if as_json:
        _print_json(
            [
                {
                    "id": int(e.id),
                    "source_path": e.source_path,
                    "date": e.date,
                    "ingested_at": e.ingested_at,
                    "total_chunks": int(e.total_chunks),
                    "preview": e.preview,
                }
                for e in entries
            ]
        )
        return 0

    for e in entries:
        sys.stdout.write(f"{e.date}\t{e.source_path}\n")
    return 0


def _unique_source_path(*, date: str, source_hint: str | None) -> str:
    """
    One SQLite row per ingest. Never reuse `typed_{date}` alone — that collides
    when the user saves multiple entries on the same day.
    """
    uid = uuid.uuid4().hex[:12]
    if source_hint and source_hint.strip():
        stem = Path(source_hint.strip()).stem or "entry"
        safe = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:48] or "entry"
        return f"photo_{safe}_{uid}"
    return f"typed_{date}_{uid}"


def cmd_ingest_text(text: str, date: str, as_json: bool, source_hint: str | None) -> int:
    conn = connect()
    init_db(conn)

    source_path = _unique_source_path(date=date, source_hint=source_hint)
    raw_path = save_entry_text(date=date, source_path=source_path, text=text)
    preview = (text.strip().replace("\n", " ")[:220]).strip()
    word_count = len([w for w in text.split() if w.strip()])

    entry_id = upsert_entry(
        conn,
        date=date,
        source_path=source_path,
        raw_text_path=str(raw_path),
        word_count=word_count,
        preview=preview,
        total_chunks=0,
    )

    # Vectorize and store chunks in ChromaDB
    errors: list[str] = []
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    chunks_stored = 0
    skipped = 0

    try:
        import chromadb  # type: ignore
    except Exception as e:
        errors.append(
            "chromadb not available in this Python environment. "
            "Use the bundled venv (src-tauri/python) or install chromadb.\n"
            f"Import error: {e}"
        )
        chunks = []

    if chunks:
        try:
            embedder = default_embedder()
            client = chromadb.PersistentClient(path=str(chroma_dir()))
            collection = client.get_or_create_collection(
                name="journal_chunks",
                metadata={"hnsw:space": "cosine"},
            )

            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            ids = [f"{entry_id}:{c.index}" for c in chunks]
            metadatas = [
                {
                    "entry_id": int(entry_id),
                    "date": date,
                    "source_path": source_path,
                    "chunk_index": int(c.index),
                    "raw_text_path": str(raw_path),
                }
                for c in chunks
            ]
            # Upsert so re-ingesting same entry replaces vectors
            collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
            chunks_stored = len(chunks)
            # Persist chunk count for UI/streak/insights without scanning Chroma.
            upsert_entry(
                conn,
                date=date,
                source_path=source_path,
                raw_text_path=str(raw_path),
                word_count=word_count,
                preview=preview,
                total_chunks=int(chunks_stored),
            )
        except EmbeddingError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Vector DB ingest failed: {e}")

    report = {
        "source_path": source_path,
        "date": date,
        "chunks_stored": int(chunks_stored),
        "skipped": int(skipped),
        "errors": errors,
    }

    if as_json:
        _print_json(report)
    else:
        sys.stdout.write("OK\n")
    return 0


def cmd_query(text: str, top_k: int, as_json: bool) -> int:
    errors: list[str] = []
    try:
        import chromadb  # type: ignore
    except Exception as e:
        errors.append(f"chromadb import failed: {e}")
        if as_json:
            _print_json({"results": [], "errors": errors})
        return 2

    try:
        embedder = default_embedder()
        client = chromadb.PersistentClient(path=str(chroma_dir()))
        collection = client.get_or_create_collection(name="journal_chunks", metadata={"hnsw:space": "cosine"})
        q_emb = embedder.embed([text])[0]
        res = collection.query(query_embeddings=[q_emb], n_results=int(top_k), include=["documents", "metadatas", "distances"])
        # chroma returns lists-of-lists
        results = []
        for doc, meta, dist in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0]):
            results.append({"text": doc, "metadata": meta, "distance": dist})
        if as_json:
            _print_json({"results": results, "errors": errors})
        else:
            for r in results:
                sys.stdout.write(r["text"] + "\n\n")
        return 0
    except Exception as e:
        errors.append(str(e))
        if as_json:
            _print_json({"results": [], "errors": errors})
        return 2


def _delete_chroma_for_entry_id(entry_id: int) -> list[str]:
    errors: list[str] = []
    try:
        import chromadb  # type: ignore
    except Exception as e:
        return [f"chromadb import: {e}"]
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir()))
        collection = client.get_or_create_collection(
            name="journal_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        ids: list[str] = []
        for wh in (
            {"entry_id": entry_id},
            {"entry_id": str(entry_id)},
            {"$and": [{"entry_id": {"$eq": entry_id}}]},
        ):
            try:
                got = collection.get(where=wh, include=[])
                ids = got.get("ids") or []
            except Exception:
                continue
            if ids:
                break
        if ids:
            collection.delete(ids=ids)
    except Exception as e:
        errors.append(str(e))
    return errors


def cmd_delete_entry_by_id(entry_id: int, as_json: bool) -> int:
    conn = connect()
    init_db(conn)
    row = get_entry_by_id(conn, entry_id)
    if row is None:
        if as_json:
            _print_json({"ok": False, "error": "not_found", "errors": []})
            return 0
        sys.stderr.write("not found\n")
        return 1
    eid = int(row["id"])
    chroma_errors = _delete_chroma_for_entry_id(eid)
    deleted = delete_entry_row_and_file_by_id(conn, entry_id=eid)
    if deleted is None:
        if as_json:
            _print_json({"ok": False, "error": "not_found", "errors": chroma_errors})
            return 0
        return 1
    if as_json:
        _print_json({"ok": True, "deleted_id": eid, "errors": chroma_errors})
        return 0
    sys.stdout.write("OK\n")
    return 0


def cmd_delete_entry(date: str, source_path: str, as_json: bool) -> int:
    conn = connect()
    init_db(conn)
    row = get_entry(conn, date=date, source_path=source_path)
    if row is None:
        if as_json:
            _print_json({"ok": False, "error": "not_found", "errors": []})
            return 0
        sys.stderr.write("not found\n")
        return 1
    entry_id = int(row["id"])
    chroma_errors = _delete_chroma_for_entry_id(entry_id)
    deleted = delete_entry_row_and_file(conn, date=date, source_path=source_path)
    if deleted is None:
        if as_json:
            _print_json({"ok": False, "error": "not_found", "errors": chroma_errors})
            return 0
        sys.stderr.write("not found\n")
        return 1
    if as_json:
        _print_json({"ok": True, "deleted_id": entry_id, "errors": chroma_errors})
        return 0
    sys.stdout.write("OK\n")
    return 0


def cmd_read_entry_by_id(entry_id: int, as_json: bool) -> int:
    conn = connect()
    init_db(conn)
    row = get_entry_by_id(conn, entry_id)
    if row is None:
        if as_json:
            _print_json(
                {
                    "ok": False,
                    "error": "not_found",
                    "text": "",
                    "date": "",
                    "source_path": "",
                    "errors": [],
                }
            )
            return 0
        sys.stderr.write("not found\n")
        return 1
    date = str(row["date"])
    source_path = str(row["source_path"])
    return _read_entry_from_row(row, date=date, source_path=source_path, as_json=as_json)


def _read_entry_from_row(
    row: dict,
    *,
    date: str,
    source_path: str,
    as_json: bool,
) -> int:
    raw = Path(str(row["raw_text_path"]))
    if not raw.is_file():
        if as_json:
            _print_json(
                {
                    "ok": False,
                    "error": "file_missing",
                    "text": "",
                    "date": date,
                    "source_path": source_path,
                    "errors": [],
                }
            )
            return 0
        sys.stderr.write("file missing\n")
        return 1
    try:
        text = raw.read_text(encoding="utf-8")
    except OSError as e:
        if as_json:
            _print_json(
                {
                    "ok": False,
                    "error": f"read_failed: {e}",
                    "text": "",
                    "date": date,
                    "source_path": source_path,
                    "errors": [],
                }
            )
            return 0
        return 1
    if as_json:
        _print_json(
            {
                "ok": True,
                "text": text,
                "date": date,
                "source_path": source_path,
                "word_count": int(row["word_count"]),
                "errors": [],
            }
        )
        return 0
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_read_entry(date: str, source_path: str, as_json: bool) -> int:
    conn = connect()
    init_db(conn)
    row = get_entry(conn, date=date, source_path=source_path)
    if row is None:
        if as_json:
            _print_json(
                {
                    "ok": False,
                    "error": "not_found",
                    "text": "",
                    "date": date,
                    "source_path": source_path,
                    "errors": [],
                }
            )
            return 0
        sys.stderr.write("not found\n")
        return 1
    return _read_entry_from_row(row, date=date, source_path=source_path, as_json=as_json)


def cmd_count_entries(as_json: bool) -> int:
    conn = connect()
    init_db(conn)
    row = conn.execute("SELECT COUNT(1) AS n FROM entries").fetchone()
    n = int(row["n"]) if row is not None else 0
    if as_json:
        _print_json({"entry_count": n, "errors": []})
    else:
        sys.stdout.write(str(n) + "\n")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="librarian.py")
    p.add_argument("--json", action="store_true", help="Output JSON")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-entries", action="store_true")
    g.add_argument("--ingest-text", type=str, default=None)
    g.add_argument("--query", type=str, default=None, help="Semantic query against stored chunks")
    g.add_argument("--count-entries", action="store_true", help="Fast SQLite-only entry count (for health checks)")
    g.add_argument(
        "--delete-entry",
        action="store_true",
        help="Remove one entry by --date and --source (SQLite, file, Chroma chunks)",
    )
    g.add_argument(
        "--read-entry",
        action="store_true",
        help="Print full text for one entry (--date and --source required)",
    )

    p.add_argument("--date", type=str, default="")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional hint for this entry (e.g. original image path). Used to build a unique id; "
        "multiple typed saves on the same day always get distinct ids.",
    )
    p.add_argument(
        "--entry-id",
        type=int,
        default=None,
        dest="entry_id",
        help="Stable SQLite row id (preferred for read/delete from the UI).",
    )
    p.add_argument("--top-k", type=int, default=3)

    args = p.parse_args(argv)

    if args.list_entries:
        return cmd_list_entries(as_json=args.json)

    if args.ingest_text is not None:
        if not args.date:
            if args.json:
                _print_json({"error": "missing --date"})
            else:
                sys.stderr.write("missing --date\n")
            return 2
        return cmd_ingest_text(
            text=args.ingest_text,
            date=args.date,
            as_json=args.json,
            source_hint=args.source,
        )

    if args.query is not None:
        return cmd_query(text=args.query, top_k=args.top_k, as_json=args.json)

    if args.count_entries:
        return cmd_count_entries(as_json=args.json)

    if args.delete_entry:
        if args.entry_id is not None:
            return cmd_delete_entry_by_id(entry_id=args.entry_id, as_json=args.json)
        if not args.date or not args.source:
            if args.json:
                _print_json({"ok": False, "error": "missing --date or --source or --entry-id"})
            else:
                sys.stderr.write("missing --date or --source or --entry-id\n")
            return 2
        return cmd_delete_entry(date=args.date, source_path=args.source, as_json=args.json)

    if args.read_entry:
        if args.entry_id is not None:
            return cmd_read_entry_by_id(entry_id=args.entry_id, as_json=args.json)
        if not args.date or not args.source:
            if args.json:
                _print_json({"ok": False, "error": "missing --date or --source or --entry-id", "text": ""})
            else:
                sys.stderr.write("missing --date or --source or --entry-id\n")
            return 2
        return cmd_read_entry(date=args.date, source_path=args.source, as_json=args.json)

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

