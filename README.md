# Journal Buddy

A **private, local-first** journaling app for macOS. Write or import handwritten pages, search your library, and chat with **Sage** — an on-device companion powered by **Ollama** (`llama3:8b`). Your journal text and AI conversations stay on your Mac.

> **Beta:** This build is for testing. Not a substitute for professional mental health care.

---

## Download (beta testers)

1. Open the [GitHub Releases](https://github.com/DevarshShroff/Local_Journal_Buddy/releases) page for this project.
2. Download the macOS **`.dmg`** for your chip:
   - **Apple Silicon (M1/M2/M3):** `Journal Buddy_*_aarch64.dmg`
   - **Intel:** `Journal Buddy_*_x64.dmg` (if published)
3. Open the DMG, drag **Journal Buddy** into **Applications**, and launch from there.

**First launch:** macOS may show “unidentified developer.” Right-click the app → **Open** → **Open** again.

---

## Where the build file lives (for maintainers)

After a release build:

```bash
./bundle_python.sh   # once per machine / when Python deps change
npm run build
```

| Artifact | Path |
|----------|------|
| **DMG (upload this to GitHub Releases)** | `src-tauri/target/release/bundle/dmg/Journal Buddy_1.0.1_aarch64.dmg` |
| **`.app` bundle** | `src-tauri/target/release/bundle/macos/Journal Buddy.app` |

Version and architecture in the filename follow `productName`, `version` in `src-tauri/tauri.conf.json`, and your Mac’s target triple. The DMG is **gitignored** — attach it manually to each GitHub Release.

**Expected size:** After `./bundle_python.sh`, the Python tree is ~**1–1.5 GB**. The release `.app` / `.dmg` should be **hundreds of MB** (compressed), not ~5 MB. A tiny DMG means `python/` was not bundled — see troubleshooting below.

**Verify the bundle before uploading:**

```bash
du -sh "src-tauri/target/release/bundle/macos/Journal Buddy.app/Contents/Resources/python"
# Should report ~1G+, not "No such file"
```

---

## How to use

### 1. First-time setup (onboarding)

When you open the app, a short checklist runs:

| Check | What it means |
|-------|----------------|
| **Python environment** | Bundled interpreter for OCR and search |
| **Ollama server** | Local AI engine (installed automatically on first run) |
| **Llama3:8b model** | Downloaded once (~4.5 GB); needs network the first time |
| **Journal database** | SQLite + vector index on your Mac |

Wait until all show **✓**, then click **Continue**. You can **Skip for now** if you only want to browse (chat may be limited until setup finishes).

Optional: enable **Password protect** to set a device-local lock (stored in the app’s secure storage area, not sent online).

### 2. Write a new entry

1. Go to **New Entry**.
2. Pick a **date**, type your journal text.
3. Click **Save Entry**.

Each save is indexed automatically so Sage can find it later.

### 3. Import photos (single or batch)

**Single photo (New Entry → Photo mode)**

1. Drop or choose a `.jpg` / `.png` / `.heic` image.
2. Run **Extract text** (macOS Vision OCR).
3. Review the text, then **Save**.

**Many photos (Batch Import)**

1. Open **Batch Import**.
2. Use **Choose folder…** (recommended on desktop) and select a folder of journal images.
3. Choose date mode: from **filename/EXIF** or one **default date** for the whole folder.
4. Wait for the progress bar; entries appear in the library when done.

### 4. Library & insights

- **Library** — search, open, delete entries; **Reflect** sends an entry to Sage (full text goes to the model; the chat bubble stays short).
- **Insights** — streak, counts, and a 6-month activity heatmap.

### 5. Chat with Sage

1. Open **Chat with Sage**.
2. Ask anything about your journal (themes, patterns, gentle reflection).
3. Sage uses **semantic search** over your saved entries plus **this session’s** conversation history.

If answers feel empty, add or re-save entries so they get indexed. Replies may show **date chips** when journal excerpts were used.

---

## Where your data is stored

Everything stays under Application Support:

```text
~/Library/Application Support/JournalBuddy/
├── journal.sqlite3    # metadata & previews
├── entries/           # plain-text copies by date
└── chroma/            # vector search index
```

Managed Ollama (if used) is separate:

```text
~/Library/Application Support/journal-buddy-ollama/
```

**Earlier beta builds** may have used `SovereignJournal` as the folder name. You can rename that folder to `JournalBuddy` or set `JOURNAL_BUDDY_DATA_DIR` to the old path (see [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)).

---

## Requirements

- **macOS 13+** (Ventura or later)
- **~8 GB free disk** for the AI model (first run)
- **Internet** only for first-time Ollama/model download (unless you use your own Ollama install)

---

## Build from source (developers)

**Prerequisites:** Node.js, Rust, Xcode CLI tools, Homebrew **Python 3.12** (for bundling).

```bash
git clone <your-repo-url>
cd Local_Journal_Buddy

npm install
./bundle_python.sh
npm run dev          # development
npm run build        # release .app + .dmg
```

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for architecture, modules, and environment variables.

### Optional environment variables

| Variable | Effect |
|----------|--------|
| `JOURNAL_BUDDY_USE_SYSTEM_OLLAMA=1` | Use existing Ollama on port `11434` instead of bundled |
| `JOURNAL_BUDDY_MANAGED_OLLAMA=0` | Disable auto-download of Ollama |
| `JOURNAL_BUDDY_DATA_DIR` | Custom data directory |

---

## Privacy

- Journal content and embeddings are stored **locally**.
- Sage runs via **Ollama on localhost** — not a cloud chat API in the default setup.
- Optional app password is **local-only** (browser storage in the desktop webview).

Do not commit personal journals, `.dmg` builds, or `src-tauri/python/` into git.

---

## Troubleshooting (beta)

| Issue | Try |
|-------|-----|
| Stuck on “installing llama3:8b” | Wait for download; check disk space and network; restart app |
| Sage says no journal context | Save at least one entry; wait a few seconds after save |
| OCR empty | Use clearer photos; macOS Vision works best on readable handwriting/print |
| “Unidentified developer” | Right-click → Open |
| Chat works but library empty | You may be in browser preview — use the **installed .app** |
| DMG only ~5 MB | `tauri.conf.json` must list `python/` under `bundle.resources`; run `./bundle_python.sh` then `npm run build` again |

---

## License

Add your license here before public release (e.g. MIT). Until then, beta builds are provided as-is for testing.

---

## Contributing

Issues and feedback welcome on GitHub. Please **do not** attach private journal screenshots or full entry text to public issues.
