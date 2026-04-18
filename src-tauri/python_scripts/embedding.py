from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class EmbeddingError(RuntimeError):
    pass


@dataclass
class Embedder:
    model_name_or_path: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__(model_name_or_path=model_name_or_path)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise EmbeddingError(
                "sentence-transformers not available in this Python environment. "
                "Use the bundled venv (src-tauri/python) or install dependencies.\n"
                f"Import error: {e}"
            ) from e

        # Reduce noisy stdout in CLI-mode.
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Ensure we stay offline if the cache exists.
        # If the model is missing, SentenceTransformer will try to download; that’s fine in dev.
        self._model = SentenceTransformer(model_name_or_path)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Convert to plain python lists so chromadb can serialize.
        vecs = self._model.encode(list(texts), normalize_embeddings=True).tolist()
        return [list(map(float, v)) for v in vecs]


_DEFAULT_EMBEDDER: SentenceTransformerEmbedder | None = None


def default_embedder() -> SentenceTransformerEmbedder:
    """
    Prefer the checked-in local model snapshot so we work offline.
    Falls back to model name (which may download in dev).
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER

    override = os.environ.get("SOVEREIGNJOURNAL_EMBED_MODEL")
    if override:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder(override)
        return _DEFAULT_EMBEDDER

    # Checked-in cache path (repo): src-tauri/python/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>
    repo_root = Path(__file__).resolve().parents[1]  # .../src-tauri
    snapshots_dir = (
        repo_root
        / "python"
        / "models"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    if snapshots_dir.exists():
        # Pick the first snapshot directory.
        for child in snapshots_dir.iterdir():
            if child.is_dir():
                _DEFAULT_EMBEDDER = SentenceTransformerEmbedder(str(child))
                return _DEFAULT_EMBEDDER

    _DEFAULT_EMBEDDER = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    return _DEFAULT_EMBEDDER

