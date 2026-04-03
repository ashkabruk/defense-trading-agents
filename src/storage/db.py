from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterator

from src.models.core import Event, TradeProposal


class SQLiteRepository:
    """Synchronous SQLite repository for events, proposals, and positions."""

    def __init__(self, db_path: str | Path = "data/trading.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    body TEXT NOT NULL,
                    url TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    tickers_json TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    importance_score REAL NOT NULL,
                    raw_data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_url_hash
                ON events(url, content_hash);

                CREATE TABLE IF NOT EXISTS trade_proposals (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    conviction REAL NOT NULL,
                    entry_rationale TEXT NOT NULL,
                    risk_factors_json TEXT NOT NULL,
                    position_size_pct REAL NOT NULL,
                    stop_loss_pct REAL NOT NULL,
                    take_profit_pct REAL NOT NULL,
                    max_holding_days INTEGER NOT NULL,
                    source_events_json TEXT NOT NULL,
                    agent_votes_json TEXT NOT NULL,
                    ragas_score REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    max_close_time TEXT NOT NULL,
                    closed_time TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(proposal_id) REFERENCES trade_proposals(id)
                );
                """
            )

    def _content_hash(self, event: Event) -> str:
        return sha256(f"{event.headline}|{event.body}".encode("utf-8")).hexdigest()

    def event_exists(self, event: Event) -> bool:
        """Check dedup key (url + content hash)."""
        content_hash = self._content_hash(event)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE url = ? AND content_hash = ? LIMIT 1",
                (event.url, content_hash),
            ).fetchone()
        return row is not None

    def save_event(self, event: Event) -> bool:
        """Insert an event if not duplicate; returns True when inserted."""
        content_hash = self._content_hash(event)
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO events (
                        id, source, timestamp, headline, body, url, content_hash,
                        entities_json, tickers_json, sentiment_score, importance_score, raw_data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.id,
                        event.source,
                        event.timestamp.isoformat(),
                        event.headline,
                        event.body,
                        event.url,
                        content_hash,
                        json.dumps(event.entities),
                        json.dumps(event.tickers),
                        event.sentiment_score,
                        event.importance_score,
                        json.dumps(event.raw_data),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def save_trade_proposal(self, proposal: TradeProposal) -> None:
        """Upsert a trade proposal by ID."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_proposals (
                    id, timestamp, ticker, direction, conviction, entry_rationale,
                    risk_factors_json, position_size_pct, stop_loss_pct, take_profit_pct,
                    max_holding_days, source_events_json, agent_votes_json, ragas_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    conviction = excluded.conviction,
                    entry_rationale = excluded.entry_rationale,
                    risk_factors_json = excluded.risk_factors_json,
                    position_size_pct = excluded.position_size_pct,
                    stop_loss_pct = excluded.stop_loss_pct,
                    take_profit_pct = excluded.take_profit_pct,
                    max_holding_days = excluded.max_holding_days,
                    source_events_json = excluded.source_events_json,
                    agent_votes_json = excluded.agent_votes_json,
                    ragas_score = excluded.ragas_score,
                    status = excluded.status
                """,
                (
                    proposal.id,
                    proposal.timestamp.isoformat(),
                    proposal.ticker,
                    proposal.direction,
                    proposal.conviction,
                    proposal.entry_rationale,
                    json.dumps(proposal.risk_factors),
                    proposal.position_size_pct,
                    proposal.stop_loss_pct,
                    proposal.take_profit_pct,
                    proposal.max_holding_days,
                    json.dumps(proposal.source_events),
                    json.dumps({k: v.model_dump() for k, v in proposal.agent_votes.items()}),
                    proposal.ragas_score,
                    proposal.status,
                ),
            )

    def fetch_recent_events(self, limit: int = 50) -> list[Event]:
        """Fetch most recent events ordered by timestamp desc."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        events: list[Event] = []
        for row in rows:
            events.append(
                Event(
                    id=row["id"],
                    source=row["source"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    headline=row["headline"],
                    body=row["body"],
                    url=row["url"],
                    entities=json.loads(row["entities_json"]),
                    tickers=json.loads(row["tickers_json"]),
                    sentiment_score=row["sentiment_score"],
                    importance_score=row["importance_score"],
                    raw_data=json.loads(row["raw_data_json"]),
                )
            )
        return events

    def fetch_contract_history(self, ticker: str, months_back: int = 12) -> list[Event]:
        """Return events for a ticker where source is contract-related."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE source IN ('sam_gov', 'defense_gov', 'dsca_arms_sales')
                  AND tickers_json LIKE ?
                  AND timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
                """,
                (f'%"{ticker}"%', f"-{months_back} months"),
            ).fetchall()

        return [
            Event(
                id=row["id"],
                source=row["source"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                headline=row["headline"],
                body=row["body"],
                url=row["url"],
                entities=json.loads(row["entities_json"]),
                tickers=json.loads(row["tickers_json"]),
                sentiment_score=row["sentiment_score"],
                importance_score=row["importance_score"],
                raw_data=json.loads(row["raw_data_json"]),
            )
            for row in rows
        ]

    def get_event_count(self) -> int:
        """Get total count of events in database."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        return row["cnt"] if row else 0
