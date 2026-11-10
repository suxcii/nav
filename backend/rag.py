"""
Unofficial Guide RAG, running fully on the DGX.

  question -> (expand) -> MiniLM embedding -> top-k chunks -> Ollama Llama -> grounded answer + sources

No cloud calls: embeddings are local (sentence-transformers) and the answer is written
by the same Ollama model the map search uses.
"""
import json
import os
import re
import threading

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(HERE, "rag_data", "chunks.json")      # made by ingest.py
CACHE_FILE = os.path.join(HERE, "rag_data", "embeddings.npy")    # made automatically on first run

MODEL = os.environ.get("RAG_MODEL", "llama3.1:8b")               # any Ollama model works here
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8                                                        # was 5 in the standalone guide
MIN_SIM = float(os.environ.get("RAG_MIN_SIM", "0.15"))           # below this, nothing relevant was found

REFUSAL = "I don't have enough information on that in my documents."

# Fixes the "work experience" -> CPT failure from the eval: add the words the documents actually use.
EXPAND = [
    (r"work experience|job|work while|work permit|work on campus|intern", "internship CPT Curricular Practical Training"),
    (r"airport|get to campus|get to sfbu|from sfo|from sjc|from oak", "airport BART Uber transportation"),
    (r"mental|therapy|counsel|stress|anxious|depress", "TimelyCare counseling wellness"),
    (r"dorm|housing|apartment|move in|bring", "residential life housing amenities"),
]

SYSTEM = """You are a helpful assistant for new students at San Francisco Bay University (SFBU).

Your job is to answer student questions using ONLY the information provided in the documents below.

Rules you must follow:
1. Only use information from the provided documents to answer. Do not use your general knowledge.
2. If the documents don't contain enough information to answer the question, say exactly: "I don't have enough information on that in my documents."
3. Do not write a sources list. The app adds it.
4. Keep your answer short, clear and easy to understand for a new student.
5. Never make up information that isn't in the documents."""

_S = {}
_lock = threading.Lock()


def load():
    """Load chunks + embeddings once (first call, or at server startup)."""
    with _lock:
        if _S:
            return _S
        from sentence_transformers import SentenceTransformer
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            chunks = json.load(f)
        model = SentenceTransformer(EMBED_MODEL)        # uses the GPU automatically if there is one
        if os.path.exists(CACHE_FILE) and len(np.load(CACHE_FILE)) == len(chunks):
            vecs = np.load(CACHE_FILE)
        else:
            vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
            np.save(CACHE_FILE, vecs)
        _S.update(chunks=chunks, vecs=np.asarray(vecs, dtype=np.float32), model=model)
        print(f"[guide] {len(chunks)} chunks ready, answering with {MODEL}")
        return _S


def expand(question):
    extra = [words for pat, words in EXPAND if re.search(pat, question, re.I)]
    return question + (" " + " ".join(extra) if extra else "")


_STOP = set("the a an and or of to in on for is are do does how what when where can i my me you your it at be with from about get".split())
_failed = False


def keyword_retrieve(question, top_k=TOP_K):
    """Safety net: no AI model needed. Scores chunks by how many question words they contain."""
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    words = {w for w in re.findall(r"[a-z0-9]+", expand(question).lower()) if len(w) > 2 and w not in _STOP}
    scored = []
    for c in chunks:
        text = c["text"].lower()
        score = sum(1 for w in words if w in text)
        if score:
            scored.append({**c, "score": score / max(len(words), 1)})
    return sorted(scored, key=lambda c: -c["score"])[:top_k]


def retrieve(question, top_k=TOP_K):
    global _failed
    if not _failed:
        try:
            S = load()
        except Exception as e:
            print("[guide] embedding model not available, using keyword search instead:", e)
            _failed = True
    if _failed:
        return keyword_retrieve(question, top_k)
    q = S["model"].encode([expand(question)], normalize_embeddings=True)[0]
    sims = S["vecs"] @ q                                  # cosine similarity (vectors are normalized)
    idx = np.argsort(-sims)[:top_k]
    return [{**S["chunks"][i], "score": float(sims[i])} for i in idx]


def generate(question, chunks):
    context = "\n\n---\n\n".join(f"[Document {i + 1}: {c['source']}]\n{c['text']}" for i, c in enumerate(chunks))
    user = (f"Here are the relevant documents:\n\n{context}\n\n---\n\n"
            f"Student question: {question}\n\nAnswer using only the documents above.")
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    }, timeout=120)
    return r.json()["message"]["content"].strip()


def ask(question):
    chunks = retrieve(question)
    if not chunks or (not _failed and chunks[0]["score"] < MIN_SIM):
        return {"answer": REFUSAL, "sources": []}
    try:
        answer = generate(question, chunks)
    except Exception as e:                                   # Ollama not running / model missing
        print("[guide] AI model not reachable, showing passages instead:", e)
        top = chunks[:3]
        text = "\n\n".join("- " + re.sub(r"\s+", " ", c["text"]).strip() for c in top)
        return {"answer": "(AI model is offline, so here are the closest passages from the guide.)\n\n" + text,
                "sources": list(dict.fromkeys(c["source"] for c in top))}
    answer = re.sub(r"\n*\s*\**Sources?:?\**.*$", "", answer, flags=re.I | re.S).strip()   # we add sources ourselves
    if "don't have enough information" in answer.lower():
        return {"answer": REFUSAL, "sources": []}
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":          # quick terminal test:  python -m backend.rag
    for q in ["How do I get from SFO airport to SFBU campus?",
              "Is there free mental health support for students?",
              "What is the last day to drop a class in Fall 2025?",
              "What does on-campus housing come with and what do I need to bring?",
              "How can an international student get work experience while studying at SFBU?",
              "What is the best pizza place in New York?"]:
        res = ask(q)
        print(f"\nQ: {q}\nA: {res['answer']}\nSources: {res['sources']}")
