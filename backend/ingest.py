"""
ingest.py — Document Ingestion and Chunking Pipeline
SFBU Unofficial Guide RAG Project

This script:
1. Loads all documents from the documents/ folder (.txt and .pdf)
2. Cleans the text (removes junk, extra whitespace, page numbers)
3. Splits into 500-character chunks with 100-character overlap
4. Prints 5 sample chunks so you can verify quality
5. Saves all chunks to a JSON file for the next step (embedding)
"""

import os
import json
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# For PDF extraction
try:
    import pdfplumber
except ImportError:
    print("pdfplumber not found. Run: pip install pdfplumber")
    exit()


# ── STEP 1: LOAD DOCUMENTS ──────────────────────────────────────────────────

def load_documents(folder):
    """
    Loads all .txt and .pdf files from the documents/ folder.
    Returns a list of dicts: {text, source}
    """
    documents = []

    if not os.path.exists(folder):
        print(f"ERROR: '{folder}' folder not found. Make sure your documents are in a folder called 'documents'.")
        return documents

    files = os.listdir(folder)
    print(f"Found {len(files)} files in '{folder}/':\n")

    for filename in files:
        filepath = os.path.join(folder, filename)

        # Skip hidden files like .gitkeep
        if filename.startswith("."):
            continue

        text = ""

        # Load PDF files
        if filename.endswith(".pdf"):
            try:
                with pdfplumber.open(filepath) as pdf:
                    pages = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages.append(page_text)
                    text = "\n\n".join(pages)
                print(f"  ✅ Loaded PDF: {filename} ({len(text)} characters)")
            except Exception as e:
                print(f"  ❌ Failed to load PDF {filename}: {e}")
                continue

        # Load .txt files
        elif filename.endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"  ✅ Loaded TXT: {filename} ({len(text)} characters)")
            except Exception as e:
                print(f"  ❌ Failed to load TXT {filename}: {e}")
                continue

        else:
            print(f"  ⏭️  Skipped: {filename} (not a .txt or .pdf)")
            continue

        if text.strip():
            documents.append({
                "text": text,
                "source": filename
            })

    print(f"\nSuccessfully loaded {len(documents)} documents.\n")
    return documents


# ── STEP 2: CLEAN TEXT ───────────────────────────────────────────────────────

def clean_text(text):
    """
    Cleans raw document text:
    - Removes HTML tags and entities
    - Removes repeated page numbers
    - Removes excessive whitespace and blank lines
    - Keeps the actual content
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove HTML entities like &amp; &nbsp; &#39;
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove standalone page numbers (e.g. just "16" on a line by itself)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple spaces into one
    text = re.sub(r" +", " ", text)

    # Collapse more than 2 newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── STEP 3: CHUNK TEXT ───────────────────────────────────────────────────────

def chunk_text(text, source, chunk_size=500, overlap=100):
    """
    Splits text into chunks of chunk_size characters,
    with overlap characters of overlap between chunks.
    Attaches the source filename as metadata.
    Returns a list of dicts: {text, source, chunk_index}
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only keep chunks with real content (not empty or just whitespace)
        if len(chunk) > 50:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_index": chunk_index
            })
            chunk_index += 1

        # Move forward by (chunk_size - overlap) to create the overlap
        start += chunk_size - overlap

    return chunks


# ── STEP 4: RUN THE PIPELINE ─────────────────────────────────────────────────

def run_pipeline(folder=os.path.join(HERE, "rag_data", "documents"),
                 output_file=os.path.join(HERE, "rag_data", "chunks.json")):
    """
    Full pipeline: load → clean → chunk → save
    """
    print("=" * 60)
    print("SFBU RAG PIPELINE — Document Ingestion & Chunking")
    print("=" * 60 + "\n")

    # Load all documents
    documents = load_documents(folder)

    if not documents:
        print("No documents loaded. Check your documents/ folder.")
        return

    # Clean and chunk each document
    all_chunks = []

    print("Cleaning and chunking documents...\n")
    for doc in documents:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned, doc["source"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # ── STEP 5: PRINT 5 SAMPLE CHUNKS FOR INSPECTION ────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS — Read these and check they look good")
    print("=" * 60)

    # Pick 5 evenly spaced chunks to sample
    step = max(1, len(all_chunks) // 5)
    sample_indices = [i * step for i in range(5)]

    for i, idx in enumerate(sample_indices):
        chunk = all_chunks[idx]
        print(f"\n--- Sample Chunk {i+1} (from: {chunk['source']}) ---")
        print(chunk["text"])
        print()

    # ── STEP 6: SAVE CHUNKS TO JSON ──────────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"✅ Done! {len(all_chunks)} chunks saved to '{output_file}'")
    print("=" * 60)
    print("\nNext step: just start the server. It embeds the chunks on first run.")
    # delete the old embedding cache so it is rebuilt from the new chunks
    cache = os.path.join(HERE, "rag_data", "embeddings.npy")
    if os.path.exists(cache):
        os.remove(cache)

    # Warn if chunk count seems off
    if len(all_chunks) < 50:
        print("\n⚠️  WARNING: Fewer than 50 chunks. Your chunks might be too large.")
        print("   Consider reducing chunk_size in chunk_text().")
    elif len(all_chunks) > 2000:
        print("\n⚠️  WARNING: More than 2000 chunks. Your chunks might be too small.")
        print("   Consider increasing chunk_size in chunk_text().")
    else:
        print(f"\n✅ Chunk count looks good ({len(all_chunks)} chunks).")


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()