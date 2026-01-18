import os
import requests

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"

def hf_emotion_predict(text: str) -> dict:
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL")

    if not token:
        raise RuntimeError("HF_TOKEN is missing")
    if not model:
        raise RuntimeError("HF_MODEL is missing")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text}

    r = requests.post(f"{HF_API_URL}{model}", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # HF sometimes returns {"error": "..."}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])

    # Sometimes returns [[{label,score},...]]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        data = data[0]

    if not isinstance(data, list) or len(data) == 0:
        return {"label": "unknown", "score": 0.0, "raw": data}

    best = max(data, key=lambda x: x.get("score", 0))
    return {"label": best.get("label", "unknown"), "score": float(best.get("score", 0.0))}
