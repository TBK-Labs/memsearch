"""The provider factory must forward runtime knobs only where they are accepted."""

from __future__ import annotations

import sys
import types

import pytest

from memsearch import embeddings


def _register_stub(monkeypatch, module_path: str, class_name: str, *, accepts_runtime: bool) -> dict:
    captured: dict = {}

    if accepts_runtime:

        class _Stub:
            def __init__(self, *, model=None, batch_size=0, threads=0, max_concurrent=0, **rest) -> None:
                captured.update(
                    model=model, batch_size=batch_size, threads=threads, max_concurrent=max_concurrent, **rest
                )
    else:

        class _Stub:
            def __init__(self, *, model=None, batch_size=0, **rest) -> None:
                captured.update(model=model, batch_size=batch_size, **rest)

    module = types.ModuleType(module_path)
    setattr(module, class_name, _Stub)
    monkeypatch.setitem(sys.modules, module_path, module)
    return captured


@pytest.mark.parametrize(
    ("name", "module_path", "class_name"),
    [
        ("onnx", "memsearch.embeddings.onnx", "OnnxEmbedding"),
        ("local", "memsearch.embeddings.local", "LocalEmbedding"),
    ],
)
def test_local_providers_receive_the_runtime_knobs(monkeypatch, name, module_path, class_name) -> None:
    captured = _register_stub(monkeypatch, module_path, class_name, accepts_runtime=True)

    embeddings.get_provider(name, threads=2, max_concurrent=3)

    assert captured["threads"] == 2
    assert captured["max_concurrent"] == 3


def test_api_providers_do_not_receive_the_runtime_knobs(monkeypatch) -> None:
    captured = _register_stub(monkeypatch, "memsearch.embeddings.openai", "OpenAIEmbedding", accepts_runtime=False)

    # Would raise TypeError if the factory forwarded threads/max_concurrent.
    embeddings.get_provider("openai", threads=2, max_concurrent=3)

    assert "threads" not in captured
    assert "max_concurrent" not in captured


def test_runtime_knobs_default_to_auto(monkeypatch) -> None:
    captured = _register_stub(monkeypatch, "memsearch.embeddings.onnx", "OnnxEmbedding", accepts_runtime=True)

    embeddings.get_provider("onnx")

    assert captured["threads"] == 0
    assert captured["max_concurrent"] == 0
