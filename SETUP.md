# Local_Journal_Buddy — Setup Guide
## Module A: OCR Engine

> **Platform:** macOS only (Apple Silicon or Intel) · **Python:** 3.10+

---

## 1 · Prerequisites

| Tool | Min Version | Install |
|------|-------------|---------|
| macOS | 13 Ventura+ | — |
| Xcode CLI Tools | latest | `xcode-select --install` |
| Homebrew | latest | https://brew.sh |
| Python (Homebrew) | 3.10+ | `brew install python` |

---

## 2 · One-Time Setup

### 2.1 — Verify Homebrew Python
```bash
# Confirm you are using Homebrew Python, not system Python
which python3
# Expected: /opt/homebrew/bin/python3  (Apple Silicon)
#        or /usr/local/bin/python3     (Intel)

python3 --version
# Expected: Python 3.10.x or higher
```

### 2.2 — Create a virtual environment
```bash
# From the root of your project folder
cd ~/Projects/Local_Journal_Buddy   # adjust to your path

python3 -m venv .venv
source .venv/bin/activate

# Your prompt should now show (.venv)
```

> **Tip:** Add `.venv/` to your `.gitignore`.

### 2.3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

pyobjc is a large package (~60 MB). Installation typically takes 1–3 minutes.

### 2.4 — Verify the install
```bash
python3 - <<'EOF'
from Vision import VNRecognizeTextRequest
print("✓ Vision framework accessible")
EOF
```

Expected output:
```
✓ Vision framework accessible
```

---

## 3 · Project Structure

```
Local_Journal_Buddy/
├── .venv/                   # virtual environment (git-ignored)
├── ocr_engine.py            # Module A  ← you are here
├── requirements.txt
├── journals/                # drop your journal photos here
│   ├── page_001.jpg
│   ├── page_002.heic
│   └── ...
└── output/                  # extracted .txt files land here
```

---

## 4 · Running the OCR Engine

### Single image
```bash
python3 ocr_engine.py --image journals/page_001.heic
```

### Batch a whole folder (recommended)
```bash
python3 ocr_engine.py --folder journals/
```

### Batch with a custom output directory
```bash
python3 ocr_engine.py --folder journals/ --output output/
```

### Batch and get JSON (for piping into Module B)
```bash
python3 ocr_engine.py --folder journals/ --output output/ --json
```

### Batch without saving .txt files (Module B ingests directly)
```bash
python3 ocr_engine.py --folder journals/ --no-save --json
```

---

## 5 · Using as a Python Module

```python
from ocr_engine import ocr_image, ocr_folder

# Single image
result = ocr_image("journals/page_001.heic")
print(result.text)
print(f"Confidence: {result.confidence_avg:.0%}")

# Batch folder — ready for Module B
results = ocr_folder("journals/", save_txt=False)
for entry in results:
    print(entry.source_path, "→", entry.word_count, "words")
```

The `OCRResult` object exposes:

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | `str` | Absolute path of the original image |
| `text` | `str` | Full extracted text |
| `timestamp` | `str` | ISO-8601 time of OCR run |
| `word_count` | `int` | Quick quality signal |
| `confidence_avg` | `float` | Average Vision confidence (0–1) |
| `txt_path` | `str \| None` | Path of saved .txt (if `save_txt=True`) |
| `errors` | `list[str]` | Non-empty if something went wrong |

---

## 6 · Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'Vision'` | pyobjc not installed in active env | `pip install pyobjc-framework-Vision` |
| `ImportError` on Cocoa / CoreImage | Missing sub-frameworks | `pip install pyobjc-framework-Cocoa pyobjc-framework-CoreImage` |
| `which python3` shows `/usr/bin/python3` | Using system Python | `source .venv/bin/activate` |
| `.heic` file fails silently | macOS < 13 may need HEIC codec | Upgrade to macOS 13+ or convert with `sips -s format jpeg input.heic --out output.jpg` |
| Empty text on a clear photo | Vision confidence too low | Try better lighting / higher resolution scan |
| `Permission denied` on output dir | Directory write permissions | `mkdir -p output && chmod 755 output` |

---

## 7 · macOS Privacy Permissions

On first run macOS may prompt for **Photos** or **Files and Folders** access if you point the script at `~/Pictures` or another protected folder.

- Go to **System Settings → Privacy & Security → Files and Folders**
- Grant access to **Terminal** (or your IDE) for the folders you need.

No network permissions are required — everything runs 100% offline.

---

## 8 · Next Steps

Once Module A is working, the `OCRResult` list feeds directly into:

- **Module B** (Librarian) — `ocr_folder(..., save_txt=False)` → ChromaDB ingestion
- **Module C** (Brain) — Module B retrieves context → Ollama/Llama3 generates response
- **Module D** (Tauri UI) — calls the Python pipeline via Tauri sidecar commands