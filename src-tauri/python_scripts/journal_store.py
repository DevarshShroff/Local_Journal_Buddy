from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_DIR_NAME = ".journal_buddy"
STORE_FILE_NAME = "store.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def store_path() -> Path:
    override = os.environ.get("JOURNAL_BUDDY_DATA_DIR")
    if override:
        return Path(override) / STORE_FILE_NAME

    home = Path(os.path.expanduser("~"))
    preferred = home / APP_DIR_NAME
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred / STORE_FILE_NAME
    except PermissionError:
        tmp = Path(tempfile.gettempdir()) / "journal_buddy"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp / STORE_FILE_NAME


def load_store() -> dict[str, Any]:
    p = store_path()
    if not p.exists():
        return {"entries": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # If the file is corrupted, don't crash the app; start fresh.
        return {"entries": []}


def save_store(store: dict[str, Any]) -> None:
    p = store_path()
    payload = json.dumps(store, ensure_ascii=False, indent=2) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload, encoding="utf-8")


@dataclass
class Entry:
    source_path: str
    date: str
    ingested_at: str
    text: str


def list_entries(store: dict[str, Any]) -> list[Entry]:
    raw = store.get("entries", [])
    out: list[Entry] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        out.append(
            Entry(
                source_path=str(e.get("source_path", "")),
                date=str(e.get("date", "")),
                ingested_at=str(e.get("ingested_at", "")),
                text=str(e.get("text", "")),
            )
        )
    return out


def upsert_entry(store: dict[str, Any], source_path: str, date: str, text: str) -> None:
    entries = store.setdefault("entries", [])
    if not isinstance(entries, list):
        store["entries"] = []
        entries = store["entries"]

    now = utc_now_iso()
    for e in entries:
        if isinstance(e, dict) and e.get("source_path") == source_path and e.get("date") == date:
            e["text"] = text
            e["ingested_at"] = now
            return

    entries.append(
        {"source_path": source_path, "date": date, "ingested_at": now, "text": text}
    )
