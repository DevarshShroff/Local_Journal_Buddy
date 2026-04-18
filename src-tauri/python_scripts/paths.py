from __future__ import annotations

import os
from pathlib import Path


def app_support_dir(app_name: str = "SovereignJournal") -> Path:
    """
    Returns ~/Library/Application Support/<app_name> on macOS.
    Falls back to ~/.<app_name> if the macOS path isn't available.
    """
    home = Path(os.path.expanduser("~"))
    mac = home / "Library" / "Application Support" / app_name
    return mac


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

