"""Local embedding via sentence-transformers (runs on CPU/GPU).

Requires: ``pip install 'memsearch[local]'`` or ``uv add 'memsearch[local]'``
No API key needed.
"""

from __future__ import annotations

import asyncio
from functools import partial


def _detect_device() -> str:
    """Detect best available device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalEmbedding:
    """sentence-transformers embedding provider."""

    _DEFAULT_BATCH_SIZE = 512

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        batch_size: int = 0,
        threads: int = 0,
        max_concurrent: int = 0,
    ) -> None:
        import io
        import os
        import sys

        from .runtime import acquire_model_slot, resolve_max_concurrent, resolve_threads

        # Held for as long as the model is resident — see runtime.acquire_model_slot.
        self._slot = acquire_model_slot(resolve_max_concurrent("local", max_concurrent))

        # Suppress noisy "Loading weights" tqdm bar and safetensors LOAD REPORT
        prev_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            import torch
            from sentence_transformers import SentenceTransformer

            # Same bound as the ONNX provider: torch also defaults its intra-op
            # pool to the core count, which is wrong for one of many processes.
            torch.set_num_threads(resolve_threads(threads))
            self._st_model = SentenceTransformer(model, device=_detect_device(), trust_remote_code=True)
        finally:
            sys.stderr = old_stderr
            if prev_tqdm is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm
        self._model = model
        self._dimension = self._st_model.get_sentence_embedding_dimension() or 384
        self._batch_size = batch_size if batch_size > 0 else self._DEFAULT_BATCH_SIZE

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def close(self) -> None:
        """Release the resident-model slot.  Safe to call more than once."""
        slot = getattr(self, "_slot", None)
        if slot is not None:
            slot.release()
            self._slot = None

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.close()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from .utils import batched_embed

        return await batched_embed(texts, self._embed_batch, self._batch_size)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            partial(self._st_model.encode, texts, normalize_embeddings=True),
        )
        return embeddings.tolist()
