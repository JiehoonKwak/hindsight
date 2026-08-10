"""Regression coverage for runtime text-search reconciliation."""

from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from hindsight_api import migrations


class _Result:
    def __init__(self, *, scalar=None, row=None):
        self._scalar = scalar
        self._row = row

    def scalar(self):
        return self._scalar

    def fetchone(self):
        return self._row


@dataclass(frozen=True)
class _TableState:
    column: str | None
    index: str | None
    rows: int
    indexdef: str = ""


class _Connection:
    def __init__(self, tables: dict[str, _TableState]):
        self.tables = tables
        self.statements: list[str] = []
        self.table_checks: list[str] = []
        self.commits = 0

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.tables" in sql:
            table_name = params["table_name"]
            self.table_checks.append(table_name)
            return _Result(scalar=table_name in self.tables)
        if "information_schema.columns" in sql:
            column_type = self.tables[params["table_name"]].column
            return _Result(row=("USER-DEFINED", column_type) if column_type else None)
        if "FROM pg_indexes" in sql:
            table = self.tables[params["table_name"]]
            return _Result(row=(table.index, table.indexdef) if table.index else None)
        if sql.lstrip().startswith("SELECT COUNT(*)"):
            table_name = sql.split(".", 1)[1].split()[0]
            return _Result(scalar=self.tables[table_name].rows)
        return _Result()

    def commit(self):
        self.commits += 1


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connect(self):
        yield self.conn


def _run(monkeypatch, tables: dict[str, _TableState], extension="pgroonga"):
    conn = _Connection(tables)
    monkeypatch.setattr(migrations, "create_engine", lambda *args, **kwargs: _Engine(conn))
    migrations.ensure_text_search_extension("postgresql://unused", text_search_extension=extension)
    return conn


def _normalized_statements(conn):
    return [" ".join(statement.split()) for statement in conn.statements]


def test_populated_legacy_mental_models_adds_pgroonga_without_losing_native_rollback(monkeypatch):
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="text", index="pgroonga", rows=20),
            "mental_models": _TableState(column="tsvector", index="gin", rows=5),
        },
    )
    statements = _normalized_statements(conn)

    assert conn.table_checks == ["memory_units", "mental_models"]
    assert any(
        "ALTER INDEX public.idx_mental_models_text_search RENAME TO idx_mental_models_text_search_native" in statement
        for statement in statements
    )
    assert not any("DROP COLUMN" in statement and "mental_models" in statement for statement in statements)
    assert any(
        "CREATE INDEX idx_mental_models_text_search ON public.mental_models "
        "USING pgroonga ((COALESCE(name, '') || ' ' || content))" in statement
        for statement in statements
    )
    assert not any("DELETE FROM" in statement for statement in statements)
    assert conn.commits == 1


def test_dual_projection_pgroonga_state_is_idempotent(monkeypatch):
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="text", index="pgroonga", rows=20),
            "mental_models": _TableState(column="tsvector", index="pgroonga", rows=5),
        },
    )

    assert conn.table_checks == ["memory_units", "mental_models"]
    assert conn.commits == 0
    assert not any(statement.lstrip().startswith(("ALTER", "CREATE", "DROP")) for statement in conn.statements)


def test_populated_memory_units_backend_switch_remains_fail_closed(monkeypatch):
    conn = _Connection(
        {
            "memory_units": _TableState(column="tsvector", index="gin", rows=20),
            "mental_models": _TableState(column="tsvector", index="gin", rows=5),
        }
    )
    monkeypatch.setattr(migrations, "create_engine", lambda *args, **kwargs: _Engine(conn))

    with pytest.raises(RuntimeError, match=r"memory_units\(20 rows\)"):
        migrations.ensure_text_search_extension("postgresql://unused", text_search_extension="pgroonga")

    assert conn.commits == 0
    assert not any(statement.lstrip().startswith(("ALTER", "CREATE", "DROP")) for statement in conn.statements)


def test_populated_unknown_mental_model_index_remains_fail_closed(monkeypatch):
    conn = _Connection(
        {
            "memory_units": _TableState(column="text", index="pgroonga", rows=20),
            "mental_models": _TableState(column="tsvector", index="bm25", rows=5),
        }
    )
    monkeypatch.setattr(migrations, "create_engine", lambda *args, **kwargs: _Engine(conn))

    with pytest.raises(RuntimeError, match=r"mental_models\(5 rows\)"):
        migrations.ensure_text_search_extension("postgresql://unused", text_search_extension="pgroonga")

    assert conn.commits == 0
    assert not any(statement.lstrip().startswith(("ALTER", "CREATE", "DROP")) for statement in conn.statements)


def test_empty_mental_models_native_reconcile_restores_generated_projection(monkeypatch):
    conn = _run(
        monkeypatch,
        {
            "memory_units": _TableState(column="tsvector", index="gin", rows=0),
            "mental_models": _TableState(column="text", index="pgroonga", rows=0),
        },
        extension="native",
    )
    statements = _normalized_statements(conn)

    assert any(
        "ADD COLUMN search_vector tsvector GENERATED ALWAYS AS" in statement and "COALESCE(name, '')" in statement
        for statement in statements
    )
    assert conn.commits == 1
