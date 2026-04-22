#!/bin/bash
# bundle_python.sh — Bundles Python + all deps into src-tauri/python/
# Run this ONCE before `npm run tauri build`
#
# What it does:
#   1. Creates a standalone Python 3.12 venv inside src-tauri/python/
#   2. Installs all pip deps into that venv
#   3. Copies all .py scripts into src-tauri/python/
#   4. Tauri's `resources: ["python/**"]` then bundles the whole folder
#      into JournalBuddy.app/Contents/Resources/python/
#
# Usage:
#   chmod +x bundle_python.sh
#   ./bundle_python.sh

set -euo pipefail

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; BOLD="\033[1m"; RESET="\033[0m"
ok()     { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
info()   { echo -e "     $1"; }
header() { echo -e "\n${BOLD}$1${RESET}"; echo -e "$(printf '─%.0s' {1..50})"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BUNDLE_DIR="$SCRIPT_DIR/src-tauri/python"
SCRIPTS_SRC="$SCRIPT_DIR/src-tauri/python_scripts"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

header "1 · Preparing bundle directory"
rm -rf "$PYTHON_BUNDLE_DIR"
mkdir -p "$PYTHON_BUNDLE_DIR"
ok "Clean bundle dir: $PYTHON_BUNDLE_DIR"

header "2 · Finding Homebrew Python 3.12"
PYTHON=""
for candidate in \
    "/opt/homebrew/opt/python@3.12/bin/python3.12" \
    "/opt/homebrew/bin/python3.12" \
    "/usr/local/opt/python@3.12/bin/python3.12" \
    "/opt/homebrew/bin/python3" \
    "/usr/local/bin/python3"; do
    if [ -f "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    warn "Python 3.12 not found. Install: brew install python@3.12"
    exit 1
fi
ok "Using Python: $PYTHON ($($PYTHON --version))"

header "3 · Creating isolated venv"
"$PYTHON" -m venv "$PYTHON_BUNDLE_DIR" --copies
ok "Venv created"

VENV_PIP="$PYTHON_BUNDLE_DIR/bin/pip"
VENV_PYTHON="$PYTHON_BUNDLE_DIR/bin/python3"

header "4 · Installing Python dependencies"
"$VENV_PIP" install --upgrade pip --quiet

if [ ! -f "$REQ_FILE" ]; then
    warn "requirements.txt not found at $REQ_FILE"
    exit 1
fi
info "Installing from requirements.txt…"
"$VENV_PIP" install -r "$REQ_FILE" --quiet
ok "Dependencies installed"

header "5 · Pre-downloading ML models into bundle"
info "Downloading all-MiniLM-L6-v2 embedding model (~22 MB)…"
"$VENV_PYTHON" -c "
from sentence_transformers import SentenceTransformer
import os
# Download to a local cache inside the bundle
cache = '$(echo $PYTHON_BUNDLE_DIR)/models'
os.makedirs(cache, exist_ok=True)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache
SentenceTransformer('all-MiniLM-L6-v2')
print('  Model downloaded')
"
ok "Embedding model cached"

header "6 · Copying Python scripts"
for script in \
    "ocr_engine.py" \
    "ocr_corrector.py" \
    "librarian.py" \
    "brain.py" \
    "chunker.py" \
    "embedding.py" \
    "journal_store.py" \
    "paths.py" \
    "sovereign_store.py" \
    "vision_ocr.py"; do

    src="$SCRIPTS_SRC/$script"
    if [ -f "$src" ]; then
        cp "$src" "$PYTHON_BUNDLE_DIR/$script"
        ok "$script"
    else
        warn "$script not found at $src — skipping"
    fi
done

header "7 · Notes"
info "Bundled model cache lives at: $PYTHON_BUNDLE_DIR/models"
info "The app sets HF_HOME/SENTENCE_TRANSFORMERS_HOME via Rust env for offline use."
ok "No patching needed"

header "8 · Bundle size check"
BUNDLE_SIZE=$(du -sh "$PYTHON_BUNDLE_DIR" | cut -f1)
ok "Total bundle size: $BUNDLE_SIZE"

echo ""
echo -e "${BOLD}$(printf '═%.0s' {1..50})${RESET}"
echo -e "${GREEN}${BOLD}  ✓ Python bundle ready!${RESET}"
echo -e "$(printf '═%.0s' {1..50})"
echo ""
echo -e "  Next step:  ${BOLD}npm run tauri build${RESET}"
echo ""
