"""Regression coverage for runtime text-search reconciliation."""

from contextlib import contextmanager

from hindsight_api import migrations


class _Result:
    def __init__(self, *, scalar=None, row=None):
        self._scalar = scalar
        self._row = row

    def scalar(self):
        return self._scalar

    def fetchone(self):
        return self._row


class _VChordConnection:
    def __init__(self):
        self.table_checks: list[str] = []
        self.statements: list[str] = []
        self.commits = 0

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.tables" in sql:
            self.table_checks.append(params["table_name"])
            return _Result(scalar=True)
        if "information_schema.columns" in sql:
            return _Result(row=("USER-DEFINED", "bm25vector"))
        if "FROM pg_indexes" in sql:
            return _Result(row=("bm25", "CREATE INDEX USING bm25"))
        return _Result()

    def commit(self):
        self.commits += 1


class _Engine:
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connect(self):
        yield self.conn


def test_vchord_reconciliation_targets_and_backfills_mental_models(monkeypatch):
    conn = _VChordConnection()
    monkeypatch.setattr(migrations, "create_engine", lambda *args, **kwargs: _Engine(conn))

    migrations.ensure_text_search_extension("postgresql://unused", text_search_extension="vchord")

    assert conn.table_checks == ["memory_units", "mental_models"]
    assert all("reflections" not in statement for statement in conn.statements)
    assert any(
        "UPDATE public.mental_models" in statement and "WHERE search_vector IS NULL" in statement
        for statement in conn.statements
    )
    assert conn.commits == 1
