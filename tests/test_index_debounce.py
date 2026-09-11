"""Tests for the minimum-interval debounce on ``memsearch index``."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from memsearch import cli as cli_module
from memsearch import core as core_module
from memsearch.cli import cli
from memsearch.config import MemSearchConfig
from memsearch.index_report import IndexReport
from memsearch.index_state import index_debounce_remaining

COLLECTION = "memsearch_chunks"


def _write_state(state_path: Path, *, age_seconds: float, collection: str = COLLECTION) -> None:
    last_success = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "collection": collection,
                "last_success_at": last_success.isoformat().replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# index_debounce_remaining
# ----------------------------------------------------------------------


def test_debounce_disabled_by_default(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    _write_state(state_path, age_seconds=0)

    assert index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=0) == 0.0


def test_debounce_reports_remaining_time_after_a_recent_index(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    _write_state(state_path, age_seconds=10)

    remaining = index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=60)

    assert 45 < remaining <= 50


def test_debounce_clears_once_the_interval_has_passed(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    _write_state(state_path, age_seconds=120)

    assert index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=60) == 0.0


def test_debounce_ignores_state_for_another_collection(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    _write_state(state_path, age_seconds=1, collection="other")

    assert index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=60) == 0.0


def test_debounce_allows_indexing_without_a_previous_success(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    state_path.write_text(json.dumps({"collection": COLLECTION, "status": "error"}), encoding="utf-8")

    assert index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=60) == 0.0
    assert index_debounce_remaining(None, collection=COLLECTION, min_interval_seconds=60) == 0.0


def test_debounce_ignores_an_unparsable_timestamp(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    state_path.write_text(
        json.dumps({"collection": COLLECTION, "last_success_at": "not a timestamp"}), encoding="utf-8"
    )

    assert index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=60) == 0.0


def test_debounce_accepts_an_injected_clock(tmp_path: Path) -> None:
    state_path = tmp_path / ".index-state.json"
    _write_state(state_path, age_seconds=0)

    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    remaining = index_debounce_remaining(state_path, collection=COLLECTION, min_interval_seconds=60, now=future)

    assert remaining == pytest.approx(30.0, abs=1.0)


# ----------------------------------------------------------------------
# CLI wiring
# ----------------------------------------------------------------------


def _install_fake_memsearch(monkeypatch, cfg: MemSearchConfig, calls: list[bool]) -> None:
    class FakeMemSearch:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def index_with_report(self, *, force=False):
            calls.append(force)
            return IndexReport(indexed_chunks=3, total_files=1, indexed_files=1)

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli_module, "resolve_config", lambda _overrides=None, **_kw: cfg)
    monkeypatch.setattr(core_module, "MemSearch", FakeMemSearch)


def test_cli_index_skips_when_the_last_index_is_recent(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / ".memsearch" / "memory"
    memory_dir.mkdir(parents=True)
    _write_state(tmp_path / ".memsearch" / ".index-state.json", age_seconds=5)
    cfg = MemSearchConfig()
    cfg.indexing.min_interval_seconds = 600
    calls: list[bool] = []
    _install_fake_memsearch(monkeypatch, cfg, calls)

    result = CliRunner().invoke(cli, ["index", str(memory_dir)])

    assert result.exit_code == 0
    assert "Skipped indexing" in result.output
    assert calls == []


def test_cli_index_runs_when_the_last_index_is_stale(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / ".memsearch" / "memory"
    memory_dir.mkdir(parents=True)
    _write_state(tmp_path / ".memsearch" / ".index-state.json", age_seconds=1200)
    cfg = MemSearchConfig()
    cfg.indexing.min_interval_seconds = 600
    calls: list[bool] = []
    _install_fake_memsearch(monkeypatch, cfg, calls)

    result = CliRunner().invoke(cli, ["index", str(memory_dir)])

    assert result.exit_code == 0
    assert "Indexed 3 chunks." in result.output
    assert calls == [False]


def test_cli_index_force_overrides_the_debounce(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / ".memsearch" / "memory"
    memory_dir.mkdir(parents=True)
    _write_state(tmp_path / ".memsearch" / ".index-state.json", age_seconds=5)
    cfg = MemSearchConfig()
    cfg.indexing.min_interval_seconds = 600
    calls: list[bool] = []
    _install_fake_memsearch(monkeypatch, cfg, calls)

    result = CliRunner().invoke(cli, ["index", str(memory_dir), "--force"])

    assert result.exit_code == 0
    assert "Indexed 3 chunks." in result.output
    assert calls == [True]


def test_cli_index_is_unconditional_when_the_debounce_is_off(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / ".memsearch" / "memory"
    memory_dir.mkdir(parents=True)
    _write_state(tmp_path / ".memsearch" / ".index-state.json", age_seconds=0)
    cfg = MemSearchConfig()
    calls: list[bool] = []
    _install_fake_memsearch(monkeypatch, cfg, calls)

    result = CliRunner().invoke(cli, ["index", str(memory_dir)])

    assert result.exit_code == 0
    assert calls == [False]


def test_cli_index_skip_leaves_the_previous_state_untouched(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / ".memsearch" / ".index-state.json"
    memory_dir = tmp_path / ".memsearch" / "memory"
    memory_dir.mkdir(parents=True)
    _write_state(state_path, age_seconds=5)
    before = state_path.read_text(encoding="utf-8")
    cfg = MemSearchConfig()
    cfg.indexing.min_interval_seconds = 600
    _install_fake_memsearch(monkeypatch, cfg, [])

    CliRunner().invoke(cli, ["index", str(memory_dir)])

    assert state_path.read_text(encoding="utf-8") == before
