"""In-memory Supabase table stub for unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeResult:
    data: Any


class FakeQuery:
    def __init__(self, store: "FakeStore", table: str):
        self.store = store
        self.table = table
        self._op = "select"
        self._payload: dict | None = None
        self._filters: list[tuple[str, str, Any]] = []
        self._limit: int | None = None
        self._on_conflict: str | None = None

    def select(self, *_cols: str) -> "FakeQuery":
        self._op = "select"
        return self

    def insert(self, payload: dict) -> "FakeQuery":
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload: dict) -> "FakeQuery":
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload: dict, on_conflict: str | None = None) -> "FakeQuery":
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def eq(self, key: str, value: Any) -> "FakeQuery":
        self._filters.append((key, "eq", value))
        return self

    def limit(self, n: int) -> "FakeQuery":
        self._limit = n
        return self

    def execute(self) -> FakeResult:
        rows = self.store.tables.setdefault(self.table, [])
        if self.table in self.store.fail_tables:
            raise RuntimeError(self.store.fail_tables[self.table])

        if self._op == "select":
            matched = self._match(rows)
            if self._limit is not None:
                matched = matched[: self._limit]
            return FakeResult(data=matched)

        if self._op == "update":
            matched = self._match(rows)
            for row in matched:
                row.update(self._payload or {})
            return FakeResult(data=matched)

        if self._op == "insert":
            payload = dict(self._payload or {})
            self._enforce_unique(rows, payload)
            if "id" not in payload:
                payload["id"] = f"{self.table}-{len(rows) + 1}"
            rows.append(payload)
            return FakeResult(data=[payload])

        if self._op == "upsert":
            payload = dict(self._payload or {})
            keys = [k.strip() for k in (self._on_conflict or "").split(",") if k.strip()]
            existing = None
            if keys:
                for row in rows:
                    if all(row.get(k) == payload.get(k) for k in keys):
                        existing = row
                        break
            if existing:
                existing.update(payload)
                return FakeResult(data=[existing])
            self._enforce_unique(rows, payload)
            if "id" not in payload:
                payload["id"] = f"{self.table}-{len(rows) + 1}"
            rows.append(payload)
            return FakeResult(data=[payload])

        return FakeResult(data=[])

    def _match(self, rows: list[dict]) -> list[dict]:
        matched = []
        for row in rows:
            ok = True
            for key, _op, value in self._filters:
                if row.get(key) != value:
                    ok = False
                    break
            if ok:
                matched.append(row)
        return matched

    def _enforce_unique(self, rows: list[dict], payload: dict) -> None:
        for row in rows:
            if (
                row.get("chat_id") == payload.get("chat_id")
                and row.get("workspace_id") == payload.get("workspace_id")
                and payload.get("chat_id") is not None
            ):
                raise RuntimeError(
                    "duplicate key value violates unique constraint "
                    '"chat_memory_chat_id_workspace_id_key"'
                )


@dataclass
class FakeStore:
    tables: dict[str, list[dict]] = field(default_factory=dict)
    fail_tables: dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)
