"""SQLite-backed session store for RexSeek Agent.

只存 Agent 需要的结构化状态，不存图片/视频：
- last_subject / last_objects / last_query
- history（最近 12 轮）
- image_key（防止换图后复用旧目标）
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SESSION_RE = re.compile(r"^s_[A-Za-z0-9_-]{6,64}$")


def new_session_id() -> str:
    return "s_" + uuid.uuid4().hex[:16]


class SessionStore:
    def __init__(self, path: Path, ttl_seconds: int = 7 * 24 * 3600):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL)"
        )
        self.conn.commit()

    def load(self, sid: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT data, updated FROM sessions WHERE id=?", (sid,)).fetchone()
            if not row:
                return None
            if time.time() - float(row[1]) > self.ttl:
                self.conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
                self.conn.commit()
                return None
            try:
                data = json.loads(row[0])
            except json.JSONDecodeError:
                return None
        data["updated"] = float(row[1])
        return data

    def save(self, sid: str, data: dict) -> None:
        payload = dict(data)
        now = time.time()
        payload["updated"] = now
        body = json.dumps(payload, ensure_ascii=False)
        with self.lock:
            self.conn.execute(
                "INSERT INTO sessions(id, data, updated) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
                (sid, body, now),
            )
            self.conn.commit()

    def delete(self, sid: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self.conn.commit()

    def cleanup(self) -> int:
        cutoff = time.time() - self.ttl
        with self.lock:
            cur = self.conn.execute("DELETE FROM sessions WHERE updated < ?", (cutoff,))
            self.conn.commit()
            return cur.rowcount
