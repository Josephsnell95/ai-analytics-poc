"""Log every question to SQLite: what was asked, what came back, and what it cost.

One row per question. The full transcript and every query attempt are kept as
JSON, so a wrong answer can be traced to the exact step that went wrong.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .loop import AgentResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS question_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at        TEXT NOT NULL,          -- UTC ISO timestamp
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    question        TEXT NOT NULL,
    status          TEXT NOT NULL,          -- answered, declined, no_query, capped, error
    answer          TEXT,
    assumptions     TEXT,                   -- JSON list
    semantic_query  TEXT,                   -- JSON, the last successful query
    sql             TEXT,
    row_count       INTEGER,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    latency_s       REAL NOT NULL,
    llm_calls       INTEGER NOT NULL,
    tool_calls      INTEGER NOT NULL,
    failed_queries  INTEGER NOT NULL,       -- run_query calls rejected by the compiler
    error           TEXT,
    declined_by     TEXT,                   -- guard (not_available list) or model
    unverified      TEXT,                   -- JSON list of answer figures not found in the result
    attempts        TEXT,                   -- JSON list of every query tried
    transcript      TEXT                    -- JSON list of messages
);
"""


class QuestionLog:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit can rerun a session on a different thread, so allow that.
        # Writes are short and one at a time, so this is safe for a local app.
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a log file was created, so old logs keep working."""
        have = {row[1] for row in self.con.execute("PRAGMA table_info(question_log)")}
        for column in ("declined_by", "unverified"):
            if column not in have:
                self.con.execute(f"ALTER TABLE question_log ADD COLUMN {column} TEXT")
        self.con.commit()

    def write(self, r: AgentResult) -> int:
        u = r.usage
        cur = self.con.execute(
            """INSERT INTO question_log (asked_at, provider, model, question, status, answer,
                   assumptions, semantic_query, sql, row_count, input_tokens, output_tokens,
                   cost_usd, latency_s, llm_calls, tool_calls, failed_queries, error,
                   declined_by, unverified, attempts, transcript)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                r.provider, r.model, r.question, r.status, r.answer,
                json.dumps(r.assumptions), json.dumps(r.semantic_query) if r.semantic_query else None,
                r.sql, len(r.rows) if r.semantic_query else None,
                u.input_tokens, u.output_tokens, u.cost_usd, u.latency_s, u.llm_calls, u.tool_calls,
                sum(1 for a in r.attempts if a.error), r.error, r.declined_by,
                json.dumps(r.unverified) if r.unverified else None,
                json.dumps([a.__dict__ for a in r.attempts], default=str),
                json.dumps(r.transcript, default=str),
            ),
        )
        self.con.commit()
        return cur.lastrowid

    def get(self, question_id: int | None = None) -> dict | None:
        """One logged question as a dict; the most recent if no id is given."""
        self.con.row_factory = sqlite3.Row
        sql = "SELECT * FROM question_log " + ("WHERE id = ?" if question_id else "ORDER BY id DESC LIMIT 1")
        row = self.con.execute(sql, (question_id,) if question_id else ()).fetchone()
        self.con.row_factory = None
        return dict(row) if row else None

    def summary(self) -> list[tuple]:
        """Per-model totals, the basis of the cost-per-question figure."""
        return self.con.execute(
            """SELECT provider, model, COUNT(*), SUM(status = 'answered'), SUM(status = 'declined'),
                      ROUND(AVG(latency_s), 1), ROUND(AVG(input_tokens + output_tokens)),
                      ROUND(SUM(cost_usd), 6), ROUND(AVG(cost_usd), 6)
               FROM question_log GROUP BY 1, 2 ORDER BY 1, 2"""
        ).fetchall()

    def close(self) -> None:
        self.con.close()
