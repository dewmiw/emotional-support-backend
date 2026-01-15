# mood_routes.py
from __future__ import annotations
from flask import Blueprint, jsonify, request
from pathlib import Path
import sqlite3
import datetime as dt

mood_bp = Blueprint("mood", __name__)

# ---- Storage (SQLite) -------------------------------------------------------
DB_PATH = Path(__file__).parent / "mood_daily.db"

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS moods_daily (
            date TEXT PRIMARY KEY,               -- YYYY-MM-DD (UTC)
            emotion TEXT NOT NULL,
            updated_ts TEXT NOT NULL             -- ISO time for last change
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_moods_daily_emotion ON moods_daily(emotion);")

_init_db()

# ---- Allowed emotions (categorical only; no scoring) ------------------------
EMOTIONS = [
    {"key": "anger",   "emoji": "😠", "label": "Anger"},
    {"key": "fear",    "emoji": "😨", "label": "Fear"},
    {"key": "sadness", "emoji": "😢", "label": "Sadness"},
    {"key": "surprise","emoji": "😮", "label": "Surprise"},
    {"key": "love",    "emoji": "🥰", "label": "Love"},
    {"key": "joy",     "emoji": "😊", "label": "Joy"},
]
EMO_KEYS = {e["key"] for e in EMOTIONS}

def _today_utc() -> str:
    return dt.datetime.utcnow().date().isoformat()

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _daterange_utc(days: int) -> list[str]:
    # Returns list of YYYY-MM-DD for [today-days+1 ... today], inclusive
    base = dt.datetime.utcnow().date()
    return [(base - dt.timedelta(days=d)).isoformat() for d in range(days-1, -1, -1)]

# ---- Routes -----------------------------------------------------------------

@mood_bp.get("/mood/allowed")
def mood_allowed():
    """List allowed emotion keys and emojis (for rendering the picker)."""
    return jsonify({"emotions": EMOTIONS})

@mood_bp.get("/mood/today")
def mood_today_get():
    """Return today's mood if set, else null."""
    today = _today_utc()
    with _conn() as con:
        row = con.execute(
            "SELECT date, emotion, updated_ts FROM moods_daily WHERE date = ?",
            (today,)
        ).fetchone()
    if not row:
        return jsonify({"date": today, "emotion": None, "updated_ts": None})
    return jsonify(dict(row))

@mood_bp.post("/mood/today")
def mood_today_set():
    """
    Set/replace today's mood.
    Body: { "emotion": "<key from /mood/allowed>" }
    """
    data = request.get_json(force=True) or {}
    emotion = str(data.get("emotion", "")).strip().lower()
    if emotion not in EMO_KEYS:
        return jsonify({"error": f"Invalid emotion '{emotion}'"}), 400

    today = _today_utc()
    ts = _now_iso()
    with _conn() as con:
        # Upsert: one mood per day
        con.execute("""
            INSERT INTO moods_daily (date, emotion, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                emotion = excluded.emotion,
                updated_ts = excluded.updated_ts
        """, (today, emotion, ts))

    return jsonify({"ok": True, "date": today, "emotion": emotion, "updated_ts": ts})

@mood_bp.post("/mood/set")
def mood_set_any_date():
    """
    Set/replace mood for a specific date (optional).
    Body: { "emotion": "<key>", "date": "YYYY-MM-DD" }  # date defaults to today
    """
    data = request.get_json(force=True) or {}
    emotion = str(data.get("emotion", "")).strip().lower()
    date = str(data.get("date") or _today_utc())

    # Basic YYYY-MM-DD validation
    try:
        dt.date.fromisoformat(date)
    except Exception:
        return jsonify({"error": "Invalid date; expected YYYY-MM-DD"}), 400

    if emotion not in EMO_KEYS:
        return jsonify({"error": f"Invalid emotion '{emotion}'"}), 400

    ts = _now_iso()
    with _conn() as con:
        con.execute("""
            INSERT INTO moods_daily (date, emotion, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                emotion = excluded.emotion,
                updated_ts = excluded.updated_ts
        """, (date, emotion, ts))

    return jsonify({"ok": True, "date": date, "emotion": emotion, "updated_ts": ts})

@mood_bp.get("/mood/series")
def mood_series():
    """
    Return a continuous day-by-day series for charts.
    Query: ?days=7 (default 7; allowed 7 or 30)
    Response:
      {
        "days": 7,
        "dates": ["2026-01-06", ...],
        "emotions": ["joy", null, "anger", ...],
        "y_index": [5, null, 0, ...],     # index into allowed order (for plotting)
        "legend": ["anger","fear","sadness","surprise","love","joy"]
      }
    """
    try:
        days = int(request.args.get("days", 7))
    except Exception:
        days = 7
    if days not in (7, 30):
        days = 7

    legend = [e["key"] for e in EMOTIONS]
    index_of = {k: i for i, k in enumerate(legend)}

    axis = _daterange_utc(days)
    start_date, end_date = axis[0], axis[-1]

    with _conn() as con:
        rows = con.execute("""
            SELECT date, emotion FROM moods_daily
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
        """, (start_date, end_date)).fetchall()

    by_date = {r["date"]: r["emotion"] for r in rows}

    emotions = [by_date.get(d) for d in axis]
    y_index = [index_of[e] if e in index_of else None for e in emotions]

    return jsonify({
        "days": days,
        "dates": axis,
        "emotions": emotions,   # strings or null
        "y_index": y_index,     # ints or null (plot against legend order)
        "legend": legend
    })
