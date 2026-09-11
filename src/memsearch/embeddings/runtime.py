"""Resource policy for providers that load a model into this process.

memsearch is normally run as a short-lived CLI process: the Claude Code plugin
invokes ``memsearch index`` from its Stop hook at the end of every turn, once
per project.  Nothing coordinates those processes, so on a machine with many
projects a burst of them starts at the same time and each one independently
loads a model.  Two defaults make that burst expensive:

* ONNX Runtime sizes ``intra_op_num_threads`` from the core count, so every
  process alone tries to use the whole machine.
* Each process keeps its own copy of the model resident for its whole life
  (measured at 1.74-1.88 GB RSS for the default bge-m3 int8 export).

This module holds the two bounds that keep that in check — a thread cap applied
to every ONNX session we create, and a cross-process gate on how many local
models may be resident at once.  API-backed providers need neither.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Auto thread cap.  ONNX Runtime defaults intra_op_num_threads to the core
# count, which is the wrong default for a process that is one of many: on a
# 16-core box a single `memsearch index` was measured consuming 6.3-7.1 cores of
# user CPU per wall second, and 30 concurrent ones drove load average to 315.
# Environment variables do not help — OMP_NUM_THREADS / ORT_NUM_THREADS /
# MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / NUMEXPR_NUM_THREADS /
# VECLIB_MAXIMUM_THREADS all set to 1 left parallelism unchanged, because ORT
# sizes its own intra-op pool.  Only SessionOptions.intra_op_num_threads does.
# 4 keeps single-run latency close to uncapped while leaving the machine usable.
AUTO_THREADS = 4

# Providers that hold a model in this process.  Everything else (openai,
# google, voyage, jina, mistral, ollama) sends the text to a server and holds
# no model in memory, so the resident-model gate does not apply to them.
LOCAL_MODEL_PROVIDERS = frozenset({"onnx", "local"})

# Local providers default to one resident model per machine.
AUTO_LOCAL_MAX_CONCURRENT = 1

#: Escape hatch for the resident-model gate.  Overrides configuration; ``0`` or
#: unset means "use the configured value", a negative value disables the gate.
MAX_CONCURRENT_ENV_VAR = "MEMSEARCH_EMBED_MAX_CONCURRENT"

#: How long to wait for a slot before giving up and loading anyway.
DEFAULT_ACQUIRE_TIMEOUT = 120.0

_LOCK_DIR = Path("~/.memsearch/.locks")
_LOCK_PREFIX = "embed"
_POLL_INTERVAL = 0.25

# Slots are kept alive here for the life of the process.  An flock lives on the
# open file descriptor, so letting a slot be garbage collected would close the
# fd and silently drop the lock while the model is still resident.
_held_slots: list[ModelSlot] = []
_atexit_registered = False


def resolve_threads(configured: int) -> int:
    """Return the intra-op thread count to use for a local model.

    ``configured`` wins when positive.  ``0`` (the default) means auto, which is
    deliberately capped rather than left at ONNX Runtime's default of one thread
    per core — see :data:`AUTO_THREADS`.
    """
    if configured > 0:
        return configured
    return min(AUTO_THREADS, os.cpu_count() or 1)


def onnx_session_options(threads: int):
    """Build ``onnxruntime.SessionOptions`` bounded to *threads* intra-op threads.

    onnxruntime is imported lazily so this module stays importable without the
    ``onnx`` extra installed.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    # One operator at a time: inter-op parallelism would add a second pool on
    # top of the intra-op one, which is what we are trying to bound.
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def resolve_max_concurrent(provider: str, configured: int) -> int:
    """Return how many processes may hold *provider*'s model resident at once.

    ``0`` means auto and is resolved explicitly per provider kind:

    * local-model providers (``onnx``, ``local``) auto-resolve to
      :data:`AUTO_LOCAL_MAX_CONCURRENT`, because each process keeps ~1.8 GB
      resident for its whole life;
    * API providers auto-resolve to unlimited, because they hold no model.

    A value ``<= 0`` in the return means unlimited.  The
    :data:`MAX_CONCURRENT_ENV_VAR` environment variable overrides *configured*.
    """
    override = _env_max_concurrent()
    if override is not None:
        configured = override
    if configured > 0:
        return configured
    if configured < 0:
        return 0  # explicitly disabled
    if provider in LOCAL_MODEL_PROVIDERS:
        return AUTO_LOCAL_MAX_CONCURRENT
    return 0


def _env_max_concurrent() -> int | None:
    """Read :data:`MAX_CONCURRENT_ENV_VAR`, or None when unset/auto/invalid."""
    raw = os.environ.get(MAX_CONCURRENT_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer", MAX_CONCURRENT_ENV_VAR, raw)
        return None
    return None if value == 0 else value


class ModelSlot:
    """One acquired slot of the cross-process resident-model gate.

    Held for as long as the model is resident, not just while it loads: RSS
    stays at its peak for the life of the process, so releasing after the load
    would bound nothing.
    """

    __slots__ = ("_fd", "path")

    def __init__(self, fd: int, path: Path) -> None:
        self._fd = fd
        self.path = path

    def release(self) -> None:
        """Release the slot.  Safe to call more than once."""
        if self._fd < 0:
            return
        fd, self._fd = self._fd, -1
        with contextlib.suppress(OSError):
            os.close(fd)  # closing the fd drops the flock
        with contextlib.suppress(ValueError):
            _held_slots.remove(self)


def acquire_model_slot(
    max_concurrent: int,
    *,
    timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    lock_dir: Path | str | None = None,
) -> ModelSlot | None:
    """Take one of *max_concurrent* slots before loading a model.

    Slots are ``flock`` holds on ``~/.memsearch/.locks/embed-N.lock``.  ``flock``
    is used rather than the ``O_EXCL`` pidfile idiom elsewhere in the codebase
    because the kernel releases an flock however the holder dies, so a crashed
    or OOM-killed indexer cannot wedge every other process on the machine.

    Returns the acquired slot, or ``None`` when the gate is disabled, is not
    supported on this platform, or could not be acquired within *timeout*.
    Failing to acquire is never fatal: blocking a user's session forever would
    be worse than the memory the gate is trying to save, so we warn and let the
    caller load anyway.
    """
    if max_concurrent <= 0:
        return None

    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows has no flock
        logger.debug("No fcntl.flock on this platform; resident-model gate disabled")
        return None

    directory = Path(lock_dir).expanduser() if lock_dir is not None else _LOCK_DIR.expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create lock directory %s (%s); proceeding without a slot", directory, exc)
        return None

    paths = [directory / f"{_LOCK_PREFIX}-{i}.lock" for i in range(max_concurrent)]
    deadline = time.monotonic() + max(timeout, 0.0)
    warned = False
    while True:
        for path in paths:
            slot = _try_acquire(path, fcntl)
            if slot is not None:
                return slot
        if time.monotonic() >= deadline:
            logger.warning(
                "Waited %.0fs for one of %d model slot(s) in %s; loading the model anyway",
                max(timeout, 0.0),
                max_concurrent,
                directory,
            )
            return None
        if not warned:
            warned = True
            logger.info("All %d model slot(s) busy; waiting for one to free up", max_concurrent)
        time.sleep(_POLL_INTERVAL)


def _try_acquire(path: Path, fcntl) -> ModelSlot | None:
    """Try to flock *path* without blocking; return a held slot or None."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        logger.warning("Cannot open model slot %s (%s)", path, exc)
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        return None
    slot = ModelSlot(fd, path)
    _held_slots.append(slot)
    _register_atexit()
    return slot


def _register_atexit() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_release_all)
    _atexit_registered = True


def _release_all() -> None:
    for slot in _held_slots.copy():
        slot.release()
