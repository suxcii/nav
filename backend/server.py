"""
SFBU Nav AI search backend (runs on the DGX).

Setup:
    pip install fastapi uvicorn requests
    ollama pull llama3.1:8b
Run:
    uvicorn backend.server:app --host 0.0.0.0 --port 8000
Then set USE_AI = true in index.html.
"""
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

MODEL = "llama3.1:8b"
OLLAMA = "http://localhost:11434/api/chat"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

SYSTEM = """You are a friendly campus guide. These are the places on the map:
{rooms}
Pick the ONE place that best answers the user's question.
In the answer, say which floor it is on. If it is on another floor, tell them to take the stairs or elevator.
Reply ONLY with JSON: {{"go_to": "<place id or none>", "answer": "<one short friendly sentence>"}}"""


@app.post("/ask")
def ask(body: dict):
    rooms = "\n".join(f'- {r["id"]}: {r["name"]}, {r.get("floor", "")} '
                      f'(keywords: {", ".join(r.get("keywords", []))})'
                      + (f' classes: {"; ".join(r["classes"])}' if r.get("classes") else '')
                      for r in body["rooms"])
    r = requests.post(OLLAMA, json={
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM.format(rooms=rooms)
                + (f"\nIt is now {body['now']}. If they ask about a class, use the class times." if body.get("now") else "")},
            {"role": "user", "content": body["question"]},
        ],
    }, timeout=60)
    try:
        return json.loads(r.json()["message"]["content"])
    except (KeyError, json.JSONDecodeError):
        return {"go_to": "none", "answer": "Sorry, I didn't understand. Try another question."}


# ------------------------------------------------------------------
# WHERE AM I?  Photo -> image AI (DINOv2) on the DGX -> which panorama spot you are at.
# Each panorama is cut into 12 slices; every slice gets a "fingerprint" (embedding).
# Your photo gets a fingerprint too, and the most similar slice wins.
# Setup (inside the venv):  pip install torch transformers pillow
# ------------------------------------------------------------------
import base64, glob, io, os

_W = {}   # model + index, loaded the first time someone asks


def _load():
    if _W:
        return _W
    import torch
    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _W["torch"], _W["dev"] = torch, dev
    _W["proc"] = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    _W["model"] = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    _build_index()
    return _W


def _embed(images):
    torch = _W["torch"]
    inputs = _W["proc"](images=images, return_tensors="pt").to(_W["dev"])
    with torch.no_grad():
        v = _W["model"](**inputs).pooler_output
    return torch.nn.functional.normalize(v, dim=-1)


def _slices(img, n=12):
    w, h = img.size
    vw = min(w, int(h * 1.3))            # one slice ~ what a phone photo sees
    out = []
    for i in range(n):
        x = int((w - vw) * i / (n - 1))
        out.append((img.crop((x, 0, x + vw, h)), (x + vw / 2) / w))
    return out


def _build_index():
    from PIL import Image
    keys, imgs = [], []
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "photos")
    for f in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        pid = os.path.splitext(os.path.basename(f))[0]
        img = Image.open(f).convert("RGB")
        for crop, frac in _slices(img):
            keys.append((pid, frac))
            imgs.append(crop)
    vecs = [_embed(imgs[i:i + 32]) for i in range(0, len(imgs), 32)]
    _W["keys"], _W["vecs"] = keys, _W["torch"].cat(vecs)
    print(f"[where-am-i] indexed {len(keys)} slices from {len(set(k[0] for k in keys))} panoramas on {_W['dev']}")


@app.get("/whereami/ready")
def whereami_ready():
    W = _load()
    return {"ready": True, "device": W["dev"], "slices": len(W["keys"])}


@app.post("/whereami")
def whereami(body: dict):
    from PIL import Image
    W = _load()
    raw = base64.b64decode(body["image"].split(",")[-1])
    q = _embed([Image.open(io.BytesIO(raw)).convert("RGB")])
    sims = (W["vecs"] @ q.T).squeeze(1).tolist()
    best = {}
    for (pid, frac), s in zip(W["keys"], sims):
        if pid not in best or s > best[pid][0]:
            best[pid] = (s, frac)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    pid, (score, frac) = ranked[0]
    return {"photo_id": pid, "score": round(score, 3), "view": round(frac, 3),
            "runner_up": ranked[1][0] if len(ranked) > 1 else None, "device": W["dev"]}


# ------------------------------------------------------------------
# VOICE: speech -> text with Whisper on the DGX.
# The browser sends raw microphone samples; we resample to 16 kHz and transcribe.
# ------------------------------------------------------------------
_S = {}


def _asr():
    if _S:
        return _S
    import torch
    from transformers import pipeline
    on_gpu = torch.cuda.is_available()
    _S["pipe"] = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                          device=0 if on_gpu else -1)
    _S["dev"] = "cuda" if on_gpu else "cpu"
    return _S


@app.post("/transcribe")
def transcribe(body: dict):
    import numpy as np
    S = _asr()
    audio = np.frombuffer(base64.b64decode(body["audio"]), dtype=np.float32)
    rate = int(body.get("rate", 16000))
    if rate != 16000 and len(audio):                      # simple resample to 16 kHz
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    # silence guard: Whisper invents words like "you" from silence, so skip near-silent or tiny recordings
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    print(f"[voice] {len(audio) / 16000:.1f}s of audio, loudness {rms:.4f}")
    if len(audio) < 8000 or rms < 0.003:
        return {"text": "", "device": S["dev"]}
    out = S["pipe"]({"raw": audio, "sampling_rate": 16000}, generate_kwargs={"language": "en", "task": "transcribe"})
    text = out["text"].strip()
    if text.lower().strip(" .!?,") in {"you", "thank you", "thanks", "thanks for watching", "bye"}:
        text = ""                                          # typical Whisper guesses on empty audio
    return {"text": text, "device": S["dev"]}


# ------------------------------------------------------------------
# CHAT: the Unofficial Guide (RAG over SFBU handbooks and web pages).
# Question -> top chunks (MiniLM, local) -> Ollama Llama -> grounded answer + sources.
# ------------------------------------------------------------------
import threading
from . import rag


@app.on_event("startup")
def _warm_up_guide():
    def _go():
        try:
            rag.load()                                        # embed documents in the background
        except Exception as e:
            print("[guide] warm-up failed, chat will use keyword search:", e)
    threading.Thread(target=_go, daemon=True).start()


@app.post("/chat")
def chat(body: dict):
    try:
        return rag.ask(body["question"])
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"answer": "The guide is not ready yet. Is Ollama running, and did the packages install?", "sources": []}
