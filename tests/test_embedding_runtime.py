"""Tests for the local-model runtime policy (thread cap + resident-model gate).

onnxruntime is stubbed so these run without the ``onnx`` extra installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from memsearch.embeddings import runtime

fcntl = pytest.importorskip("fcntl", reason="the resident-model gate needs fcntl.flock")


# ----------------------------------------------------------------------
# Thread cap
# ----------------------------------------------------------------------


def test_resolve_threads_uses_explicit_value() -> None:
    assert runtime.resolve_threads(1) == 1
    assert runtime.resolve_threads(8) == 8


def test_resolve_threads_auto_is_bounded_below_the_core_count(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 16)

    assert runtime.resolve_threads(0) == runtime.AUTO_THREADS
    assert runtime.resolve_threads(0) < 16


def test_resolve_threads_auto_never_exceeds_the_core_count(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 2)

    assert runtime.resolve_threads(0) == 2


def test_resolve_threads_auto_survives_unknown_core_count(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: None)

    assert runtime.resolve_threads(0) == 1


# ----------------------------------------------------------------------
# Session options
# ----------------------------------------------------------------------


class _StubSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = -1
        self.inter_op_num_threads = -1
        self.execution_mode = None


class _StubExecutionMode:
    ORT_SEQUENTIAL = "sequential"
    ORT_PARALLEL = "parallel"


class _StubOrt:
    SessionOptions = _StubSessionOptions
    ExecutionMode = _StubExecutionMode


def test_onnx_session_options_bounds_both_thread_pools(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", _StubOrt())

    options = runtime.onnx_session_options(3)

    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == _StubExecutionMode.ORT_SEQUENTIAL


def test_onnx_session_options_does_not_import_onnxruntime_at_module_import() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    module_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("onnxruntime" in line for line in module_level)


# ----------------------------------------------------------------------
# max_concurrent resolution
# ----------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["onnx", "local"])
def test_local_providers_auto_resolve_to_one(monkeypatch, provider: str) -> None:
    monkeypatch.delenv(runtime.MAX_CONCURRENT_ENV_VAR, raising=False)

    assert runtime.resolve_max_concurrent(provider, 0) == runtime.AUTO_LOCAL_MAX_CONCURRENT == 1


@pytest.mark.parametrize("provider", ["openai", "google", "voyage", "jina", "mistral", "ollama"])
def test_api_providers_auto_resolve_to_unlimited(monkeypatch, provider: str) -> None:
    monkeypatch.delenv(runtime.MAX_CONCURRENT_ENV_VAR, raising=False)

    assert runtime.resolve_max_concurrent(provider, 0) == 0


def test_explicit_max_concurrent_wins_over_auto(monkeypatch) -> None:
    monkeypatch.delenv(runtime.MAX_CONCURRENT_ENV_VAR, raising=False)

    assert runtime.resolve_max_concurrent("onnx", 4) == 4
    assert runtime.resolve_max_concurrent("openai", 4) == 4


def test_env_var_overrides_configured_value(monkeypatch) -> None:
    monkeypatch.setenv(runtime.MAX_CONCURRENT_ENV_VAR, "3")

    assert runtime.resolve_max_concurrent("onnx", 1) == 3


def test_env_var_zero_means_auto(monkeypatch) -> None:
    monkeypatch.setenv(runtime.MAX_CONCURRENT_ENV_VAR, "0")

    assert runtime.resolve_max_concurrent("onnx", 0) == 1
    assert runtime.resolve_max_concurrent("onnx", 5) == 5


def test_negative_value_disables_the_gate(monkeypatch) -> None:
    monkeypatch.setenv(runtime.MAX_CONCURRENT_ENV_VAR, "-1")

    assert runtime.resolve_max_concurrent("onnx", 0) == 0


def test_invalid_env_var_falls_back_to_configured_value(monkeypatch) -> None:
    monkeypatch.setenv(runtime.MAX_CONCURRENT_ENV_VAR, "lots")

    assert runtime.resolve_max_concurrent("onnx", 0) == 1


# ----------------------------------------------------------------------
# Resident-model gate
# ----------------------------------------------------------------------


def test_gate_is_a_no_op_when_unlimited(tmp_path: Path) -> None:
    assert runtime.acquire_model_slot(0, lock_dir=tmp_path) is None


def test_second_acquirer_blocks_while_the_first_holds_the_only_slot(tmp_path: Path) -> None:
    # flock is held on the open file description, so a second open() contends
    # even from the same process — the same way a second CLI process would.
    first = runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path)
    assert first is not None
    assert first.path.name == "embed-0.lock"

    # Fail-open: the second acquirer gives up rather than wedging the caller.
    assert runtime.acquire_model_slot(1, timeout=0.1, lock_dir=tmp_path) is None

    first.release()

    second = runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path)
    assert second is not None
    assert second.path == first.path
    second.release()


def test_slots_are_handed_out_until_they_run_out(tmp_path: Path) -> None:
    held = [runtime.acquire_model_slot(2, timeout=0, lock_dir=tmp_path) for _ in range(2)]

    assert all(slot is not None for slot in held)
    assert {slot.path.name for slot in held} == {"embed-0.lock", "embed-1.lock"}
    assert runtime.acquire_model_slot(2, timeout=0.1, lock_dir=tmp_path) is None

    for slot in held:
        slot.release()


def test_release_is_idempotent_and_untracks_the_slot(tmp_path: Path) -> None:
    slot = runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path)
    assert slot is not None
    assert slot in runtime._held_slots

    slot.release()
    slot.release()

    assert slot not in runtime._held_slots


def test_gate_serialises_two_separate_processes(tmp_path: Path) -> None:
    """A real second OS process must wait, and must get the slot once we exit."""
    script = textwrap.dedent(
        """
        import sys
        from memsearch.embeddings import runtime

        lock_dir = sys.argv[1]
        # Busy while the parent holds the only slot.
        print("busy" if runtime.acquire_model_slot(1, timeout=0, lock_dir=lock_dir) is None else "acquired")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    slot = runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path)
    assert slot is not None
    try:
        busy = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
    finally:
        slot.release()

    free = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert busy.stdout.strip() == "busy", busy.stderr
    assert free.stdout.strip() == "acquired", free.stderr


# ----------------------------------------------------------------------
# Provider integration
# ----------------------------------------------------------------------


class _StubIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _RecordingSession:
    """Mimics ort.InferenceSession; records the SessionOptions it was built with."""

    last_options = None

    def __init__(self, _model_path, sess_options=None, **_kwargs) -> None:
        import numpy as np

        type(self).last_options = sess_options
        self._np = np

    def get_outputs(self):
        return [_StubIO("dense_vecs")]

    def get_inputs(self):
        return [_StubIO("input_ids"), _StubIO("attention_mask")]

    def run(self, _output_names, feed):
        return [self._np.ones((len(feed["input_ids"]), 4), dtype=self._np.float32)]


class _StubEncoding:
    def __init__(self) -> None:
        self.ids = [1, 2, 3]
        self.attention_mask = [1, 1, 1]


class _StubTokenizer:
    @staticmethod
    def from_file(_path):
        return _StubTokenizer()

    def enable_padding(self, **_kwargs) -> None:
        pass

    def enable_truncation(self, **_kwargs) -> None:
        pass

    def encode_batch(self, texts):
        return [_StubEncoding() for _ in texts]


def _install_onnx_stubs(monkeypatch) -> None:
    ort = _StubOrt()
    ort.InferenceSession = _RecordingSession
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    hub = type(sys)("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: "unused"
    hub.list_repo_files = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    tokenizers = type(sys)("tokenizers")
    tokenizers.Tokenizer = _StubTokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)

    from memsearch.embeddings import onnx as onnx_module

    monkeypatch.setattr(
        onnx_module.OnnxEmbedding,
        "_download_model_files",
        staticmethod(lambda *_args: ("tokenizer.json", "model.onnx")),
    )


def test_onnx_provider_builds_its_session_with_a_bounded_thread_pool(monkeypatch) -> None:
    _install_onnx_stubs(monkeypatch)
    from memsearch.embeddings.onnx import OnnxEmbedding

    # max_concurrent=-1 disables the gate so this test touches no lock files.
    provider = OnnxEmbedding(threads=2, max_concurrent=-1)

    assert _RecordingSession.last_options is not None
    assert _RecordingSession.last_options.intra_op_num_threads == 2
    assert _RecordingSession.last_options.inter_op_num_threads == 1
    assert provider.close() is None


def test_onnx_provider_holds_a_slot_until_closed(monkeypatch, tmp_path: Path) -> None:
    _install_onnx_stubs(monkeypatch)
    monkeypatch.setattr(runtime, "_LOCK_DIR", tmp_path)
    from memsearch.embeddings.onnx import OnnxEmbedding

    provider = OnnxEmbedding(max_concurrent=1)
    try:
        assert provider._slot is not None
        # Still resident, so a second process must not get the slot.
        assert runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path) is None
    finally:
        provider.close()

    freed = runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path)
    assert freed is not None
    freed.release()


def test_unwritable_lock_directory_fails_open(tmp_path: Path, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", boom)

    assert runtime.acquire_model_slot(1, timeout=0, lock_dir=tmp_path / "nope") is None
