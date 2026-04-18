from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from paths import app_support_dir, ensure_dir


APP_NAME = "SovereignJournal"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def base_dir() -> Path:
    # Allow overriding for tests / dev.
    override = os.environ.get("SOVEREIGNJOURNAL_DIR")
    if override:
        return ensure_dir(Path(override))
    # In sandboxed environments we might not be allowed to create under ~/Library.
    p = app_support_dir(APP_NAME)
    try:
        return ensure_dir(p)
    except PermissionError:
        import tempfile

        return ensure_dir(Path(tempfile.gettempdir()) / APP_NAME)


def db_path() -> Path:
    return base_dir() / "journal.sqlite3"


def entries_dir() -> Path:
    return ensure_dir(base_dir() / "entries")


def chroma_dir() -> Path:
    return ensure_dir(base_dir() / "chroma")


def connect() -> sqlite3.Connection:
    p = db_path()
    ensure_dir(p.parent)
    try:
        conn = sqlite3.connect(str(p))
    except sqlite3.OperationalError:
        # Some sandboxed environments disallow writes under ~/Library.
        # Fall back to a temp directory DB.
        import tempfile

        tmp_base = ensure_dir(Path(tempfile.gettempdir()) / APP_NAME)
        tmp_db = tmp_base / "journal.sqlite3"
        conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          date TEXT NOT NULL,
          source_path TEXT NOT NULL,
          ingested_at TEXT NOT NULL,
          raw_text_path TEXT NOT NULL,
          word_count INTEGER NOT NULL,
          preview TEXT NOT NULL,
          total_chunks INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    # Migration for older DBs missing the total_chunks column.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(entries);").fetchall()}
    if "total_chunks" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN total_chunks INTEGER NOT NULL DEFAULT 0;")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_date_source
        ON entries(date, source_path);
        """
    )
    conn.commit()


def save_entry_text(*, date: str, source_path: str, text: str) -> Path:
    # Store under ~/Library/Application Support/SovereignJournal/entries/YYYY-MM-DD/<stem>.txt
    ddir = ensure_dir(entries_dir() / date)
    stem = Path(source_path).stem or "entry"
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:80] or "entry"
    p = ddir / f"{safe_stem}.txt"
    p.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return p


def upsert_entry(
    conn: sqlite3.Connection,
    *,
    date: str,
    source_path: str,
    raw_text_path: str,
    word_count: int,
    preview: str,
    total_chunks: int,
) -> int:
    now = utc_now_iso()
    cur = conn.execute(
        """
        INSERT INTO entries(date, source_path, ingested_at, raw_text_path, word_count, preview, total_chunks)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, source_path) DO UPDATE SET
          ingested_at=excluded.ingested_at,
          raw_text_path=excluded.raw_text_path,
          word_count=excluded.word_count,
          preview=excluded.preview,
          total_chunks=excluded.total_chunks
        ;
        """,
        (date, source_path, now, raw_text_path, int(word_count), preview, int(total_chunks)),
    )
    conn.commit()
    # sqlite3 lastrowid is unreliable on UPDATE; fetch row id.
    row = conn.execute(
        "SELECT id FROM entries WHERE date=? AND source_path=?",
        (date, source_path),
    ).fetchone()
    return int(row["id"]) if row else int(cur.lastrowid or 0)


@dataclass
class EntryRow:
    id: int
    source_path: str
    date: str
    ingested_at: str
    total_chunks: int
    preview: str


def list_entries(conn: sqlite3.Connection) -> list[EntryRow]:
    rows = conn.execute(
        "SELECT id, date, source_path, ingested_at, preview, total_chunks FROM entries ORDER BY date DESC, ingested_at DESC"
    ).fetchall()
    out: list[EntryRow] = []
    for r in rows:
        out.append(
            EntryRow(
                id=int(r["id"]),
                source_path=str(r["source_path"]),
                date=str(r["date"]),
                ingested_at=str(r["ingested_at"]),
                total_chunks=int(r["total_chunks"]),
                preview=str(r["preview"]),
            )
        )
    return out


def get_entry_by_id(conn: sqlite3.Connection, entry_id: int) -> dict[str, Any] | None:
    r = conn.execute("SELECT * FROM entries WHERE id=?", (int(entry_id),)).fetchone()
    return dict(r) if r else None


def get_entry(conn: sqlite3.Connection, *, date: str, source_path: str) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT * FROM entries WHERE date=? AND source_path=?",
        (date, source_path),
    ).fetchone()
    return dict(r) if r else None


def delete_entry_row_and_file(
    conn: sqlite3.Connection, *, date: str, source_path: str
) -> dict[str, Any] | None:
    """
    Remove the SQLite row and the stored .txt file. Caller should delete Chroma vectors first
    (or accept orphans) using the returned id.
    """
    row = get_entry(conn, date=date, source_path=source_path)
    if row is None:
        return None
    raw_path = Path(str(row["raw_text_path"]))
    conn.execute("DELETE FROM entries WHERE date=? AND source_path=?", (date, source_path))
    conn.commit()
    if raw_path.is_file():
        try:
            raw_path.unlink()
        except OSError:
            pass
    return {"id": int(row["id"]), "date": date, "source_path": source_path}


def delete_entry_row_and_file_by_id(conn: sqlite3.Connection, *, entry_id: int) -> dict[str, Any] | None:
    row = get_entry_by_id(conn, entry_id)
    if row is None:
        return None
    raw_path = Path(str(row["raw_text_path"]))
    conn.execute("DELETE FROM entries WHERE id=?", (int(entry_id),))
    conn.commit()
    if raw_path.is_file():
        try:
            raw_path.unlink()
        except OSError:
            pass
    return {
        "id": int(row["id"]),
        "date": str(row["date"]),
        "source_path": str(row["source_path"]),
    }

