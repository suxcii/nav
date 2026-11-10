# nav.sfbu

A 3D indoor navigation app and campus guide chat for San Francisco Bay University, powered by AI running locally on an NVIDIA DGX.

GPS stops working indoors. SFBU Nav gives students, visitors, and anyone who needs accessible routes a way to find rooms, see real photos of each space, and know which classes are on right now, all without sending any data to the cloud.

## Features

- **3D map of both floors.** Built from hand-drawn floor sketches at real scale (about 85 m x 37 m). Switch between 1F, 2F, and both floors stacked, in 2D or 3D.
- **Ask in plain English.** "I'm hungry", "where can I use a 3D printer?", "where is the Ramirez class?" A language model on the DGX picks the right room.
- **Walking routes.** Glowing dots show the shortest path around walls and through doors (A* pathfinding). Routes to the other floor go through the stairs, or the elevator if you mention a wheelchair or the elevator.
- **Turn-by-turn style directions.** "Take the stairs to the 2nd Floor, then follow the dots. Room 207 - Makerspace is on the 2nd Floor. It will be on your right, about 113 m, 1 min walk."
- **Mascot marker.** The SFBU mascot shows where you are. Drag it, or double-click any floor to move it there. After a search it walks the route, takes the stairs, and announces when it arrives.
- **Real panoramas.** Camera icons open panoramas taken at SFBU, with arrows to walk between photo spots and a mini map.
- **Class schedule.** Room labels show NOW or "in 20 min". The Classes panel lists every section (now, starting soon, later today, finished, other days) with teacher and registration status. Click a class to route to its room.
- **Chat: the Unofficial Guide.** Ask "how do I..." questions about housing, transport, deadlines, wellness, career help and where to eat. Answers come only from SFBU documents, with sources listed.
- **Where am I? (experimental).** Upload a photo and an image model on the DGX matches it against the panoramas to place you on the map.
- **Voice (experimental).** Speak your question; Whisper on the DGX transcribes it and the answer is read back out loud.

## How the DGX is used

Everything AI runs locally on the DGX. No cloud APIs, no student data leaves the machine.

| Task | Model | Runs on |
|---|---|---|
| Understand questions, pick a room | Llama 3.1 8B (Ollama) | DGX |
| Campus guide chat | MiniLM embeddings + Llama 3.1 8B (Ollama) | DGX |
| Where am I? photo matching | DINOv2 (image embeddings) | DGX |
| Speech to text | Whisper small | DGX |
| Map, routes, mascot | three.js + A* in the browser | Any device |

Measured on the DGX Spark:

| Model | Speed | Size |
|---|---|---|
| Llama 3.1 8B | 41.6 tokens/s | 5 GB |
| Llama 3.1 70B | 3.63 tokens/s | 42 GB |

We use the 8B model in the app because instant answers matter for navigation. The 70B model also runs on the same DGX, which a normal laptop cannot do.

## How it works

```
Floor sketches  ->  room list (meters)  ->  3D map in the browser (three.js)
Panoramas       ->  photo spots          ->  Pannellum viewer + auto arrows
Question        ->  backend/server.py (FastAPI)  ->  Ollama LLM on DGX  ->  room id
Room id + mascot position  ->  A* on a 0.5 m grid  ->  dotted route + directions
Photo           ->  DINOv2 on DGX        ->  best matching panorama  ->  you are here
Voice           ->  Whisper on DGX       ->  text  ->  same search as above
Chat question   ->  MiniLM embeddings    ->  matching document chunks  ->  Llama on DGX  ->  answer + sources
```

## Run it

### Map only (any computer)

The map, routes, mascot, panoramas, and class schedule work without the DGX. Search falls back to keyword matching.

```bash
cd SFBU-nav
python -m http.server 5500
```

Open http://localhost:5500. In VS Code you can also right-click `index.html` and choose Open with Live Server.

### With the AI (DGX)

```bash
# 1. AI engine and model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# 2. Python packages
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r backend/requirements.txt
# torch, transformers, pillow, numpy are only needed for Where am I? and voice

# 3. AI server (terminal 1)
uvicorn backend.server:app --host 0.0.0.0 --port 8000

# 4. Website (terminal 2)
python3 -m http.server 5500
```

In `index.html`, `USE_AI` must be `true`. Open http://localhost:5500.

No admin password? Ollama can run from your home folder: download `ollama-linux-arm64.tar.zst` from ollama.com, unpack it with `tar --zstd -xf`, and start `~/ollama/bin/ollama serve`.

## Chat: the Unofficial Guide

Answers "how do I / when is / what is" questions from SFBU handbooks and web pages, with the source files listed under each answer.

- Search bar = where is X (rooms, routes). Chat = how do I X (documents).
- Fully local: MiniLM embeddings plus the same Ollama Llama 3.1 8B. No cloud.
- Answers come only from the documents. If they only partly answer, it says what they cover. If they cover nothing, it says so instead of guessing.
- If an answer names a room on the map, a **Show on map** button appears.

Extra setup for chat (once, inside the venv):

```bash
source ~/venv/bin/activate
pip install -r backend/requirements.txt
```

The first server start embeds the documents (about a minute), then caches them in `backend/rag_data/embeddings.npy`.

Test the guide alone: `python -m backend.rag`

Change the documents: edit files in `backend/rag_data/documents/`, run `python backend/ingest.py`, then restart the server.

Use a different model: `RAG_MODEL=llama3.1:70b uvicorn backend.server:app --host 0.0.0.0 --port 8000`

## Project structure

```
SFBU-nav/
├── index.html              the whole app: map, routes, mascot, panoramas, classes, chat
├── backend/
│   ├── server.py           DGX backend: /ask (LLM), /chat (guide), /whereami (DINOv2), /transcribe (Whisper)
│   ├── rag.py              Unofficial Guide: retrieve + generate
│   ├── ingest.py           splits documents into chunks
│   ├── rag_data/           documents and chunks.json
│   └── requirements.txt    Python packages for the backend
├── assets/
│   ├── images/             mascot.png (map marker), sfbu_logo.png
│   ├── photos/             panoramas (sample placeholders + real SFBU photos)
│   └── test_photos/        photos for testing Where am I?
├── README.md
└── LICENSE
```

## Status and next steps

Working: 3D map of both floors, AI search on the DGX, routes across floors, mascot walking, panoramas, class schedule panel, Unofficial Guide chat.

Experimental: Where am I? photo matching and voice. Both are implemented but not yet tested end to end on the DGX.

Next:

- More panoramas around both floors for better photo matching
- Real course names in the schedule
- Spoken step-by-step guidance for blind and low-vision students
- Evacuation simulation on the DGX GPU

## Notes

- Floor plans are approximate, drawn by hand from walking the building.
- Class times come from SFBU Course Registration.
- Guide answers come only from the documents in `backend/rag_data/documents/`. They are not official SFBU advice.
- Some panoramas show people. Ask for permission before publishing them, or blur faces first.
- SFBU name, logo, and mascot belong to San Francisco Bay University. Used here for a student project.

## License

The code is released under the MIT License.

The license does not cover the SFBU name, logo, and mascot (`sfbu_logo.png`, `mascot.png`), which belong to San Francisco Bay University, the photos in `photos/` and `test_photos/`, or the SFBU handbooks in `backend/rag_data/documents/`. Ask before reusing them.