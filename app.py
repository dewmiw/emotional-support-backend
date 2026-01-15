# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import requests
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import sqlite3, datetime as dt
from pathlib import Path

from model.model_loader import load_emotion_model
from mood_routes import mood_bp

# ---------------------------
# Database
# ---------------------------
DB_PATH = Path(__file__).parent / "mood.db"

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _get_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS mood_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            conversation_id TEXT,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            score REAL NOT NULL
        );
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS journals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body  TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)

# ---------------------------
# App & config
# ---------------------------
torch.set_num_threads(1)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
init_db()
app.register_blueprint(mood_bp)

# Emotion classifier
emo_model, emo_tok, emo_device = load_emotion_model()
EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
MAX_LEN_EMO = 256

# Ollama chat endpoint & model
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct"

# Short rolling history per conversation_id
HISTORIES: dict[str, deque[tuple[str, str]]] = defaultdict(lambda: deque(maxlen=6))

# ---------------------------
# Helpers: emotion / reply
# ---------------------------
def detect_emotion(text: str) -> tuple[str, float]:
    """Return (label, confidence); low confidence falls back to 'neutral'."""
    enc = emo_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN_EMO)
    enc = {k: v.to(emo_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = emo_model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
    label = EMOTION_LABELS[idx] if conf >= 0.5 else "neutral"
    return label, conf

def system_text_minimal(emotion: str) -> str:
    return (
        "You are an empathetic assistant. "
        "Reply naturally and specifically to the user. "
        "Keep it concise (about 2–4 sentences). "
        f"(Tone hint: {emotion})"
    )

def gen_with_ollama_chat(system_txt: str, history, user_txt: str,
                         max_tokens: int = 160, temperature: float = 0.6) -> str:
    messages = [{"role": "system", "content": system_txt}]
    for role, txt in history:
        messages.append({"role": "user" if role == "user" else "assistant", "content": txt})
    messages.append({"role": "user", "content": user_txt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_predict": max_tokens,
            "num_ctx": 1024,
        },
        "stream": False,
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("message", {}).get("content") or "").strip()

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def clean_reply_minimal(s: str) -> str:
    for marker in ("[USER]", "[ASSISTANT]", "[SYSTEM]", "User:", "Assistant:"):
        if marker in s:
            s = s.split(marker)[0].strip()
    s = re.sub(r'\s+', ' ', s).strip()
    parts = _SENT_SPLIT.split(s)
    return " ".join(parts[:3]).strip() or s

# ---------------------------
# Helpers: Journals
# ---------------------------
def _now_iso():
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------------------
# Routes: chat
# ---------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "ollama_model": OLLAMA_MODEL,
    })

@app.post("/respond")
def respond():
    """
    Body:
      {
        "text": "...",                 # required
        "conversation_id": "abc123",   # optional
        "gen": { "max_new_tokens": 160, "temperature": 0.6 }  # optional
      }
    Returns: { "emotion": "...", "confidence": 0.xx, "reply": "..." }
    """
    data = request.get_json(force=True) or {}
    text = (data.get("text") or data.get("sentence") or "").strip()
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    conv_id = (data.get("conversation_id") or "default").strip() or "default"
    gen_overrides = data.get("gen") or {}
    max_new_tokens = int(gen_overrides.get("max_new_tokens", 160))
    temperature    = float(gen_overrides.get("temperature", 0.6))

    # 1) Detect emotion & log
    emotion, conf = detect_emotion(text)

    # 2) Build minimal system text + history
    history = HISTORIES[conv_id]
    sys_txt = system_text_minimal(emotion)

    # 3) Generate with Ollama (chat API)
    try:
        raw = gen_with_ollama_chat(sys_txt, history, text,
                                   max_tokens=max_new_tokens,
                                   temperature=temperature)
        reply = clean_reply_minimal(raw)
        if len(reply) < 2:
            reply = "I'm here."
    except Exception as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 500

    # 4) Update history
    history.append(("user", text))
    history.append(("assistant", reply))

    return jsonify({"emotion": emotion, "confidence": conf, "reply": reply})



# ---------------------------
# Journals API
# ---------------------------

@app.post("/journal")
def journal_create():
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "").strip()
    body  = (data.get("body")  or "").strip()
    if not title or not body:
        return jsonify({"error": "title and body are required"}), 400

    now = _now_iso()
    with _get_conn() as con:
        cur = con.execute(
            "INSERT INTO journals (title, body, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, body, now, now),
        )
        jid = cur.lastrowid
        row = con.execute("SELECT * FROM journals WHERE id=?", (jid,)).fetchone()

    return jsonify(dict(row)), 201


@app.get("/journal")
def journal_list():
    """Optional search: /journal?q=term"""
    q = (request.args.get("q") or "").strip()
    with _get_conn() as con:
        if q:
            rows = con.execute(
                "SELECT * FROM journals WHERE title LIKE ? OR body LIKE ? ORDER BY created_at DESC",
                (f"%{q}%", f"%{q}%"),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM journals ORDER BY created_at DESC"
            ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/journal/<int:jid>")
def journal_get(jid: int):
    with _get_conn() as con:
        row = con.execute("SELECT * FROM journals WHERE id=?", (jid,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.put("/journal/<int:jid>")
def journal_update(jid: int):
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "").strip()
    body  = (data.get("body")  or "").strip()
    if not title or not body:
        return jsonify({"error": "title and body are required"}), 400

    now = _now_iso()
    with _get_conn() as con:
        cur = con.execute(
            "UPDATE journals SET title=?, body=?, updated_at=? WHERE id=?",
            (title, body, now, jid),
        )
        if cur.rowcount == 0:
            return jsonify({"error": "not found"}), 404
        row = con.execute("SELECT * FROM journals WHERE id=?", (jid,)).fetchone()

    return jsonify(dict(row))


@app.delete("/journal/<int:jid>")
def journal_delete(jid: int):
    with _get_conn() as con:
        cur = con.execute("DELETE FROM journals WHERE id=?", (jid,))
        if cur.rowcount == 0:
            return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Run locally: http://127.0.0.1:5001/health
    app.run(debug=True, host="0.0.0.0", port=5001)
