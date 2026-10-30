"""Persistent job queue: one job per movie or episode, run one at a time.

Jobs live in SQLite next to the legacy queue, so they survive a restart. A
job that was running when the backend stopped is marked failed on startup,
never lost silently.
"""
import json
import sqlite3
import time

ACTIVE = ("queued", "running")

_COLUMNS = ("id", "op", "server", "item_id", "title", "video_path", "target_lang",
            "backend", "model", "overwrite", "params", "status", "progress",
            "message", "created_at", "updated_at")


def _connect(db_file):
    conn = sqlite3.connect(db_file, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _row(row) -> dict:
    job = dict(row)
    job["overwrite"] = bool(job["overwrite"])
    job["params"] = json.loads(job["params"] or "{}")
    return job


def init(db_file):
    with _connect(db_file) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY, op TEXT NOT NULL, server TEXT, item_id TEXT,
            title TEXT, video_path TEXT NOT NULL, target_lang TEXT NOT NULL,
            backend TEXT, model TEXT, overwrite INTEGER NOT NULL DEFAULT 0,
            params TEXT, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
            message TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
        conn.execute("""UPDATE jobs SET status = 'failed', message = 'Interrupted by a restart',
                        updated_at = ? WHERE status = 'running'""", (time.time(),))


def add(db_file, *, op, video_path, target_lang, title="", server=None, item_id=None,
        backend=None, model=None, overwrite=False, params=None) -> tuple[dict, bool]:
    """Queue a job. Returns (job, created); an identical job that is already
    queued or running is returned instead of a duplicate."""
    now = time.time()
    with _connect(db_file) as conn:
        existing = conn.execute(
            f"SELECT * FROM jobs WHERE op = ? AND video_path = ? AND target_lang = ? "
            f"AND status IN {ACTIVE}", (op, video_path, target_lang)).fetchone()
        if existing:
            return _row(existing), False
        cursor = conn.execute(
            "INSERT INTO jobs (op, server, item_id, title, video_path, target_lang, backend, model,"
            " overwrite, params, status, progress, message, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, '', ?, ?)",
            (op, server, item_id, title, video_path, target_lang, backend, model,
             int(overwrite), json.dumps(params or {}), now, now))
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _row(job), True


def claim_next(db_file, skip_backends=()) -> dict | None:
    """The oldest queued job, marked running. Jobs on a backend in
    skip_backends (e.g. Gemini at its daily limit) wait their turn."""
    with _connect(db_file) as conn:
        query, args = "SELECT * FROM jobs WHERE status = 'queued'", []
        if skip_backends:
            query += f" AND COALESCE(backend, '') NOT IN ({','.join('?' * len(skip_backends))})"
            args += list(skip_backends)
        row = conn.execute(query + " ORDER BY id LIMIT 1", args).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE jobs SET status = 'running', progress = 0, message = '', updated_at = ?"
                     " WHERE id = ?", (time.time(), row["id"]))
        return _row(conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())


def update(db_file, job_id, **fields):
    fields["updated_at"] = time.time()
    names = [name for name in fields if name in _COLUMNS]
    with _connect(db_file) as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(f'{n} = ?' for n in names)} WHERE id = ?",
                     [fields[n] for n in names] + [job_id])


def get(db_file, job_id) -> dict | None:
    with _connect(db_file) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row(row) if row else None


def recent(db_file, limit=100) -> list[dict]:
    """Active jobs first (oldest first), then the latest finished ones."""
    with _connect(db_file) as conn:
        active = conn.execute(f"SELECT * FROM jobs WHERE status IN {ACTIVE} ORDER BY id").fetchall()
        done = conn.execute(f"SELECT * FROM jobs WHERE status NOT IN {ACTIVE} ORDER BY id DESC LIMIT ?",
                            (limit,)).fetchall()
        return [_row(r) for r in active] + [_row(r) for r in done]


def active_targets(db_file) -> set[tuple[str, str]]:
    """(video_path, target_lang) of every queued or running job, for badges."""
    with _connect(db_file) as conn:
        rows = conn.execute(f"SELECT video_path, target_lang FROM jobs WHERE status IN {ACTIVE}").fetchall()
        return {(r["video_path"], r["target_lang"]) for r in rows}


def clear_finished(db_file):
    with _connect(db_file) as conn:
        conn.execute(f"DELETE FROM jobs WHERE status NOT IN {ACTIVE}")
