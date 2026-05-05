"""SQLite connection helpers and schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    """Open a connection with WAL mode and foreign keys enabled.

    When *readonly* is True the connection is opened with the immutable URI
    flag so that concurrent readers never block each other (useful inside
    DataLoader workers).
    """
    db_path = str(db_path)
    if readonly:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(db_path: str | Path) -> None:
    """Create all tables if they don't already exist."""
    schema_sql = _SCHEMA_PATH.read_text()
    conn = get_connection(db_path)
    conn.executescript(schema_sql)
    conn.close()


def reset_db(db_path: str | Path) -> None:
    """Drop and recreate the database (destructive!)."""
    p = Path(db_path)
    if p.exists():
        p.unlink()
    # Also remove WAL / SHM side-files
    for suffix in ("-wal", "-shm"):
        side = p.with_name(p.name + suffix)
        if side.exists():
            side.unlink()
    create_schema(db_path)
