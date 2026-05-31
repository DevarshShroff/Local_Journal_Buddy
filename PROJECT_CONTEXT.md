# Journal Buddy — Project Context

A single reference for **what this app is**, **why it is built this way**, **how it is structured**, and **what we implemented** during development (including fixes and design decisions from the build-out phase).

---

## 1. Product intent (psychological & ethical frame)

### What Journal Buddy is for

Journal Buddy is a **private, local-first journaling companion** on macOS (Tauri desktop). It helps someone:

- Capture life in writing or from **photos of handwritten pages**
- **Revisit** entries in a library with calendar/streak cues
- Talk to **“Sage”** — a gentle AI persona that reflects on their writing **without sending data to the cloud**

The psychological posture is deliberate:

| Principle | How the product expresses it |
|-----------|-------------------------------|
| **Safety & privacy** | Data stays on device; onboarding states this explicitly; optional **local password** gate (Web Crypto PBKDF2 + `localStorage`, not server auth). |
| **Non-clinical warmth** | Sage system prompt: listen first, short paragraphs, no preaching, “caring friend / therapist-like” tone — not diagnosis or treatment. |
| **Agency** | User can skip onboarding, skip lock, delete entries; no forced cloud signup. |
| **Continuity** | RAG over journal chunks + **chat history within a session** so follow-ups feel remembered. |
| **Gentle honesty** | If there is no journal context, the model is instructed to say so kindly and invite more writing — not hallucinate a backstory. |

### What it is *not*

- Not a replacement for professional mental health care
- Not a multi-user or synced social product
- Not dependent on OpenAI or other cloud LLM APIs for core chat (Ollama local)

### User-facing emotional design (UI)

- **Forest / sage / cream palette** — calm, botanical sidebar pattern, Lora + DM Sans typography
- **Onboarding** as a trust ritual: Python, Ollama, model, DB checks before “Continue”
- **Loading quotes** during Sage replies — pacing and reassurance while the local model runs
- **“Reflect on entry”** from the library — opens chat with the entry body in the *model* prompt but only a short label in the *visible* user bubble (privacy in the thread UI)
- **Insights** — streak, stats, GitHub-style **6-month heatmap** (activity visualization without mood taxonomy)
- Removed **mood tags / mood distribution** and noisy “typed/photo” chips to reduce labeling anxiety and clutter

---

## 2. Technical architecture (high level)

```
┌─────────────────────────────────────────────────────────────────┐
│  src/index.html  (UI: library, write, OCR, batch, chat, insights) │
│       │ invoke / listen (Tauri v2)                               │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  src-tauri/src/lib.rs          Tauri commands, events, batch OCR │
│  src-tauri/src/health.rs       Onboarding health checks          │
│  src-tauri/src/ollama_managed.rs  Bundled Ollama :11437          │
│  src-tauri/src/python.rs       Spawns bundled Python subprocesses  │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  src-tauri/python_scripts/     (copied into bundle by bundle_python.sh) │
│    librarian.py   SQLite + Chroma ingest/query/delete            │
│    brain.py       RAG + Ollama chat/generate                     │
│    embedding.py   sentence-transformers (local MiniLM)           │
│    chunker.py     Text chunks for vector index                   │
│    sovereign_store.py  Paths, SQLite schema, entry files         │
│    ocr_engine.py / vision_ocr.py / ocr_corrector.py  macOS OCR   │
│    paths.py, journal_store.py (legacy JSON store — largely superseded) │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
┌──────────────────────┐     ┌────────────────────────────────────┐
│ ~/Library/.../       │     │ Ollama @ 127.0.0.1:11437 (managed) │
│ JournalBuddy/        │     │ Model: llama3:8b (auto pull)      │
│  journal.sqlite3     │     │ or :11434 if USE_SYSTEM_OLLAMA=1  │
│  entries/YYYY-MM-DD/ │     └────────────────────────────────────┘
│  chroma/             │
└──────────────────────┘
```

### Stack

| Layer | Technology |
|-------|------------|
| Shell | **Tauri 2** (Rust), `tauri-plugin-shell` |
| UI | Single **`src/index.html`** (vanilla HTML/CSS/JS, no React) |
| AI inference | **Ollama** (`llama3:8b`), HTTP `/api/chat` or `/api/generate` |
| Retrieval | **ChromaDB** persistent client, cosine space, collection `journal_chunks` |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` (bundled snapshot under `src-tauri/python/models/` when present) |
| Structured store | **SQLite** (`journal.sqlite3`) + plain `.txt` per entry |
| OCR | **macOS Vision** via PyObjC; optional symspell / language-tool correction |
| HTTP from Rust | `reqwest` (Ollama download, health tags) |

### Privacy & data locations

- **App support dir**: `~/Library/Application Support/JournalBuddy/` (override: `JOURNAL_BUDDY_DATA_DIR`)
  - `journal.sqlite3` — metadata, previews, chunk counts
  - `entries/<date>/<stem>.txt` — full text
  - `chroma/` — vector index
- **Bundled Ollama artifacts**: app local data `journal-buddy-ollama/` (binary + `models/`)
- **Password hash**: browser `localStorage` in the webview (device-local only)
- **Git**: `src-tauri/python/` venv is **gitignored**; rebuilt via `bundle_python.sh` + `requirements.txt`

---

## 3. Project structure (repository)

```
Local_Journal_Buddy/
├── package.json              # npm scripts: dev, build
├── requirements.txt          # Python deps for bundle
├── bundle_python.sh          # Builds src-tauri/python/ venv + copies scripts
├── PROJECT_CONTEXT.md        # This document
├── src/
│   └── index.html            # Entire frontend + Tauri invoke layer
└── src-tauri/
    ├── tauri.conf.json       # App name, window, DMG, icons
    ├── Cargo.toml            # Rust deps (tauri, reqwest, tokio, …)
    ├── build.rs
    ├── capabilities/
    ├── icons/
    ├── entitlements.plist, Info.plist
    ├── src/
    │   ├── main.rs           # Entry → run()
    │   ├── lib.rs            # Commands + types
    │   ├── health.rs
    │   ├── ollama_managed.rs
    │   └── python.rs         # Resolver + subprocess env (OLLAMA_URL, HF_HOME, …)
    └── python_scripts/       # Source of truth for Python (committed)
        ├── brain.py          # Module D — Sage answers
        ├── librarian.py      # Module B — ingest, query, list, delete
        ├── sovereign_store.py
        ├── embedding.py, chunker.py
        ├── ocr_engine.py, vision_ocr.py, ocr_corrector.py
        ├── paths.py
        └── journal_store.py  # Older JSON store (not primary path)
```

### Module naming (internal lore)

- **Module B** — `librarian.py` (storage + RAG index)
- **Module D** — `brain.py` (question → retrieve → Ollama)

---

## 4. Core pipelines

### 4.1 Ingest (typed or post-OCR)

1. UI → `ingest_text` / `ingest_ocr_result` (Tauri)
2. `librarian.py --ingest-text` → unique `source_path` per save (`typed_{date}_{uuid}` or `photo_{stem}_{uuid}`)
3. Save `.txt` under `entries/`, upsert SQLite row
4. `chunk_text` (500 chars, 50 overlap) → embed → Chroma **upsert** with metadata (`entry_id`, `date`, `source_path`, …)
5. Update `total_chunks` on the row for UI/insights

### 4.2 Ask Sage (chat)

1. UI maintains **`chatHistory`** (last 12 turns, `sessionStorage`)
2. `ask_brain_stream` → `brain.py --ask --top-k 8 --history-json [...] --ollama-url <managed>`
3. `brain.py` calls `librarian.py --query` (semantic search)
4. **Retrieval fallback**: if Chroma empty or distances weak (>0.85), merge **recent SQLite entry snippets**
5. Build user message with journal excerpts + use **`/api/chat`** when history exists (multi-turn memory)
6. Rust fake-streams tokens → `brain-token` / `brain-done` (with `context_dates`, `chunks_used`)

### 4.3 Batch import

- Native folder picker (`rfd`) → walk images → Vision OCR per file → `ingest_text` with date from EXIF/filename/default
- Progress events: `batch-progress`

### 4.4 Managed Ollama (first launch)

- Default **on** in dev and release (opt out: `JOURNAL_BUDDY_MANAGED_OLLAMA=0`)
- System Ollama: `JOURNAL_BUDDY_USE_SYSTEM_OLLAMA=1` → port **11434**
- Managed: download **Ollama v0.21.0** (mac `.tgz` / win `.zip`) → serve on **11437** → `api/pull` for **llama3:8b**
- `prime_ollama_base_url()` in setup so health checks hit 11437 immediately
- UI: `ollama-setup` events + top setup banner (hidden after onboarding / model ready)
- Shutdown: only kill **child** we spawned

### 4.5 Onboarding health

Parallel checks: Python version, Ollama `/api/tags`, model name match, DB entry count via `librarian --count-entries`. Continue enabled when all required flags true.

---

## 5. Design decisions (technical)

| Decision | Rationale |
|----------|-----------|
| **Tauri + single HTML file** | Small desktop app, no bundler complexity; full offline UI. |
| **Python sidecar** | Heavy ML/OCR/Chroma ecosystem; Rust orchestrates. |
| **Dedicated Ollama port 11437** | Avoid fighting user’s existing `ollama serve` on 11434. |
| **SQLite + files + Chroma** | SQLite for list/delete/metadata; files for human-readable backup; Chroma for semantic search only. |
| **Unique source_path per ingest** | Same-day multiple typed saves must not collide on `(date, source_path)`. |
| **Stable `entry_id` in UI** | Read/delete by numeric id — avoids webview string mismatches. |
| **Tauri v2 camelCase invoke args** | `folderPath`, `imagePath`, `entryId`, `topK`, etc. |
| **Subprocess librarian from brain** | Keeps retrieval API in one module; brain stays thin. |
| **RAG + chat history** | Journal = long-term memory via vectors; chat = short-term thread via Ollama messages. |
| **No cloud LLM in core path** | Aligns with privacy promise and offline DMG story. |
| **Fake streaming** | UI expects tokens; full answer chunked in Rust until true stream exists. |
| **Don't commit `src-tauri/python/`** | Torch/transformers wheel size; CI/git friendly. |

---

## 6. Design decisions (product / UX)

| Decision | Rationale |
|----------|-----------|
| **Sage persona** | Consistent, non-judgmental reflection partner. |
| **Onboarding gate** | Surfaces misconfiguration (Python/Chroma/Ollama) before frustration in chat. |
| **Optional password lock** | Lightweight deterrence on shared machines — not enterprise security. |
| **Skip onboarding** | Power users / recovery path. |
| **Insights heatmap vs mood charts** | Show *habit* without forcing mood labels. |
| **Reflect flow** | Deep link from library to chat with full entry in model context. |
| **Setup banner only during onboarding** | Avoid permanent “installing…” clutter after model ready. |

---

## 7. What we built & fixed (chronological themes)

This section captures work from the implementation phase (not every commit).

### Platform & packaging

- Custom app icon; DMG layout in `tauri.conf.json`
- `bundle_python.sh` + `requirements.txt`; gitignore bundled venv
- macOS entitlements / Info.plist for desktop distribution

### UI / library / insights

- Batch import with native folder picker, progress events, path fixes
- Library list by SQLite id; entry viewer; delete confirm
- Sidebar botanical pattern; **bold** nav labels
- Insights: 6-month calendar heatmap; removed mood UI and tag chips
- Optional **local password** lock screen

### Ollama / onboarding

- Managed download, extract, serve, auto-pull `llama3:8b`
- Async init + `ollama-setup` events + progress banner
- Fixed: managed mode default in dev; port 11437 priming; model pull via HTTP API; early-return skipping pull; health vs wrong port 11434; banner dismiss on onboarding complete / model ready

### Chat / brain

- **Chat history** (12 turns, sessionStorage) → Ollama `/api/chat`
- **Stronger RAG**: top-8, Chroma empty/weak → recent entry fallback
- Context date chips on replies; hint when zero chunks matched
- Health check timeout increased; parallel health probes

### Data / ingest

- Unique ingest ids per entry; Chroma upsert on re-ingest
- `ingest_text` / batch wired through `librarian.py`

---

## 8. Tauri commands (API surface)

| Command | Role |
|---------|------|
| `health_check` | Onboarding status |
| `ingest_text` | Typed entry → index |
| `ingest_ocr_result` | Post-OCR save |
| `ocr_image` | Vision OCR path |
| `get_all_entries` | Library list |
| `get_journal_entry_text` | Viewer body |
| `delete_journal_entry` | SQLite + file + Chroma |
| `ask_brain` | Full JSON response |
| `ask_brain_stream` | Token events + `brain-done` metadata |
| `pick_batch_folder` | Native directory dialog |
| `batch_ingest_folder` | Folder OCR pipeline |

### Events (frontend `listen`)

| Event | Payload |
|-------|---------|
| `ollama-setup` | stage, message, pct, done |
| `brain-token` | string chunk |
| `brain-done` | `BrainResponse` |
| `batch-progress` | filename, current, total, pct, done |

---

## 9. Environment variables

| Variable | Effect |
|----------|--------|
| `JOURNAL_BUDDY_USE_SYSTEM_OLLAMA=1` | Use `http://127.0.0.1:11434`, no managed download |
| `JOURNAL_BUDDY_MANAGED_OLLAMA=0` | Disable managed bundle (use with system or external) |
| `JOURNAL_BUDDY_DATA_DIR` | Override Application Support data root |
| `JOURNAL_BUDDY_EMBED_MODEL` | Override embedding model path/name |
| `OLLAMA_URL` | Set on Python children from Rust (`effective_ollama_base()`) |
| `JOURNAL_BUDDY_OLLAMA_MODEL` | Default model in `brain.py` |

---

## 10. Build & run

```bash
# One-time (or after requirements change)
./bundle_python.sh

# Development
npm run dev
# or: cargo tauri dev

# Release .app / DMG (requires bundle_python.sh first)
npm run build
```

**Release packaging:** `src-tauri/tauri.conf.json` must include `bundle.resources: ["python/", "python_scripts/"]` so the ~1.3GB venv is copied into `Journal Buddy.app/Contents/Resources/`. Without this, the DMG stays ~5 MB and beta installs lack OCR/RAG.

Verify after build:

```bash
du -sh "src-tauri/target/release/bundle/macos/Journal Buddy.app/Contents/Resources/python"
```

---

## 11. Known limitations & future hooks

- **Chat memory** is session-scoped (`sessionStorage`), not a permanent encrypted transcript store.
- **Streaming** is simulated in Rust; true token streaming would use Ollama stream + `python::run_streaming`.
- **Linux** managed Ollama: expects system `ollama` binary; no auto-download yet.
- **Re-index** command for entries with `total_chunks == 0` is a natural follow-up if users have library rows without vectors.
- **`journal_store.py`**: legacy JSON; primary path is `sovereign_store` + `librarian`.

---

---

*Journal Buddy v1.0.1 — see [README.md](README.md) for install and beta testing.*
