# Journal Buddy

A **private, local-first** journaling app for macOS. Write or import handwritten pages, search your library, and chat with **Sage** an on-device AI companion powered by Ollama (`llama3:8b`). Your journal text and AI conversations never leave your Mac.

> **Beta:** Not a substitute for professional mental health care.

---

## Download

Grab the latest macOS `.dmg` from the **[Releases](https://github.com/DevarshShroff/Local_Journal_Buddy/releases)** page:

- **Apple Silicon (M1/M2/M3):** `Journal Buddy_*_aarch64.dmg`

Open the DMG, drag **Journal Buddy** into **Applications**, and launch it. If macOS warns "unidentified developer," right-click the app → **Open** → **Open** again.

**Requirements:** macOS 13+, ~8 GB free disk (for the AI model), internet on first run only.

---

## First-time setup

On launch, a checklist will run through: Python environment, Ollama server, Llama3:8b model (~4.5 GB download), and the local journal database. Wait for all items to show **✓**, then click **Continue**.

---

## Features

- **Write entries** — pick a date, type, and save. Entries are indexed automatically for Sage.
- **Import photos** — single or batch import of `.jpg`/`.png`/`.heic` images with OCR via macOS Vision.
- **Library** — search, open, delete, or send entries to Sage for reflection.
- **Insights** — streaks, counts, and a 6-month activity heatmap.
- **Chat with Sage** — ask about themes and patterns in your journal; Sage uses semantic search over your saved entries.

---

## Privacy

All journal content, embeddings, and AI conversations are stored locally under `~/Library/Application Support/JournalBuddy/`. Sage runs on localhost via Ollama — no cloud API involved.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Stuck on "installing llama3:8b" | Wait; check disk space and network; restart app |
| Sage has no journal context | Save at least one entry and wait a few seconds |
| OCR returns empty text | Use clearer photos; Vision works best on readable handwriting |
| DMG is only ~5 MB | Run `./bundle_python.sh` then `npm run build` again |
