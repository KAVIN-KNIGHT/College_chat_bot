"""
ingestion.py — Document ingestion pipeline.

This script reads PDF and DOCX files from the data/ directory, extracts
text page-by-page, splits it into overlapping chunks, generates
embeddings, and stores everything in a persistent ChromaDB collection.

Run this script whenever you add new documents to data/:

    python ingestion.py

It is idempotent — running it again will skip chunks that are already
stored (identified by their deterministic chunk IDs).
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import chromadb

from config import (
    DATA_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    DISTANCE_METRIC,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from embeddings import get_embeddings

# When reading DOCX files (which have no native page concept), we group
# this many paragraphs together into one logical "page" for metadata.
DOCX_PARAGRAPHS_PER_PAGE = 20


# =====================================================================
# Step 1: Extract text from documents (PDF + DOCX)
# =====================================================================

def extract_text_from_pdf(filepath: str, filename: str) -> list[dict]:
    """
    Extract text from a PDF file, page-by-page.

    Returns a list of dicts with keys: source, page, text.
    """
    pages = []
    reader = PdfReader(filepath)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Skip empty or whitespace-only pages
        if not text or not text.strip():
            continue

        pages.append({
            "source": filename,
            "page": page_num,
            "text": text.strip(),
        })

    return pages


def extract_text_from_docx(filepath: str, filename: str) -> list[dict]:
    """
    Extract text from a DOCX file.

    DOCX files don't have a built-in page structure, so we group
    paragraphs into logical "pages" of DOCX_PARAGRAPHS_PER_PAGE
    paragraphs each. This keeps metadata (page numbers) meaningful.

    Returns a list of dicts with keys: source, page, text.
    """
    pages = []
    doc = DocxDocument(filepath)

    # Collect all non-empty paragraphs
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        return pages

    # Group paragraphs into logical pages
    page_num = 1
    for i in range(0, len(paragraphs), DOCX_PARAGRAPHS_PER_PAGE):
        batch = paragraphs[i : i + DOCX_PARAGRAPHS_PER_PAGE]
        page_text = "\n".join(batch)

        pages.append({
            "source": filename,
            "page": page_num,
            "text": page_text,
        })
        page_num += 1

    return pages


def extract_text_from_documents(data_dir: str) -> list[dict]:
    """
    Read every PDF and DOCX file in `data_dir` and extract text.

    Returns a list of dicts, each representing one non-empty page:
        {
            "source": "filename.pdf",
            "page": 3,
            "text": "page text..."
        }
    """
    pages = []
    supported_extensions = (".pdf", ".docx")

    # Find all supported files in the data directory
    doc_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(supported_extensions)
    )

    if not doc_files:
        print(f"[ingestion] No PDF or DOCX files found in {data_dir}/")
        return pages

    for filename in doc_files:
        filepath = os.path.join(data_dir, filename)
        print(f"[ingestion] Reading: {filename}")

        if filename.lower().endswith(".pdf"):
            pages.extend(extract_text_from_pdf(filepath, filename))
        elif filename.lower().endswith(".docx"):
            pages.extend(extract_text_from_docx(filepath, filename))

    print(f"[ingestion] Extracted {len(pages)} non-empty pages from {len(doc_files)} document(s).")
    return pages


# =====================================================================
# Step 2: Split text into overlapping chunks
# =====================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split `text` into chunks of at most `chunk_size` characters,
    with `chunk_overlap` characters of overlap between consecutive chunks.

    Example with chunk_size=10, overlap=3:
        "ABCDEFGHIJKLMNO"
        → ["ABCDEFGHIJ", "HIJKLMNO"]
           overlap: "HIJ"

    This overlap ensures that sentences or ideas that span a chunk
    boundary still appear (at least partially) in both chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        # Move forward by (chunk_size - overlap) characters (at least 1 char)
        step = max(1, chunk_size - chunk_overlap)
        start += step

    return chunks


def create_chunks_from_pages(pages: list[dict]) -> list[dict]:
    """
    Take the page-level text and split each page into overlapping chunks.

    Returns a list of dicts:
        {
            "id":           "filename.pdf__page3__chunk2",
            "text":         "chunk content...",
            "source":       "filename.pdf",
            "page":         3,
            "chunk_index":  2
        }

    The chunk ID is deterministic so that re-running ingestion
    does not create duplicates.
    """
    all_chunks = []

    for page_info in pages:
        text_chunks = chunk_text(page_info["text"])

        for i, chunk_text_content in enumerate(text_chunks):
            # Build a deterministic, human-readable chunk ID
            chunk_id = f"{page_info['source']}__page{page_info['page']}__chunk{i}"

            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text_content,
                "source": page_info["source"],
                "page": page_info["page"],
                "chunk_index": i,
            })

    print(f"[ingestion] Created {len(all_chunks)} chunks from {len(pages)} pages.")
    return all_chunks


# =====================================================================
# Step 3: Store chunks in ChromaDB
# =====================================================================

def store_chunks_in_chromadb(chunks: list[dict]) -> None:
    """
    Store the given chunks in a persistent ChromaDB collection.

    - Connects to (or creates) the ChromaDB database at CHROMA_DB_DIR.
    - Creates (or gets) the collection named COLLECTION_NAME.
    - Skips chunks whose IDs already exist (idempotent).
    - Generates embeddings for new chunks and stores them.
    """
    if not chunks:
        print("[ingestion] No chunks to store.")
        return

    # Connect to persistent ChromaDB with cosine similarity space
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )

    # Find which chunk IDs already exist in the collection
    all_ids = [c["id"] for c in chunks]
    existing = set()

    # ChromaDB's .get() can check if IDs exist
    # We query in batches to avoid overly large requests
    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        result = collection.get(ids=batch_ids)
        existing.update(result["ids"])

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c["id"] not in existing]

    if not new_chunks:
        print("[ingestion] All chunks already exist in ChromaDB. Nothing to add.")
        return

    print(f"[ingestion] {len(existing)} chunks already exist, {len(new_chunks)} new chunks to add.")

    # Generate embeddings for the new chunks
    texts = [c["text"] for c in new_chunks]
    print(f"[ingestion] Generating embeddings for {len(texts)} chunks...")
    embeddings = get_embeddings(texts)

    # Prepare data for ChromaDB
    ids = [c["id"] for c in new_chunks]
    metadatas = [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
        }
        for c in new_chunks
    ]

    # Add to ChromaDB in batches (ChromaDB recommends batches of ~5000)
    add_batch_size = 5000
    for i in range(0, len(ids), add_batch_size):
        end = i + add_batch_size
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"[ingestion] Successfully stored {len(new_chunks)} new chunks in ChromaDB.")


# =====================================================================
# Main: Run the full ingestion pipeline
# =====================================================================

def ingest():
    """Run the complete ingestion pipeline: extract → chunk → store."""
    print("=" * 60)
    print("  College Chatbot — Document Ingestion")
    print("=" * 60)

    # Step 1: Extract text from documents (PDF + DOCX)
    pages = extract_text_from_documents(DATA_DIR)
    if not pages:
        print("\n[ingestion] No pages extracted. Add PDF or DOCX files to the data/ directory and try again.")
        return

    # Step 2: Create chunks
    chunks = create_chunks_from_pages(pages)

    # Step 3: Store in ChromaDB
    store_chunks_in_chromadb(chunks)

    print("\n[ingestion] Done! Your documents are now indexed and ready for retrieval.")
    print(f"[ingestion] ChromaDB location: {os.path.abspath(CHROMA_DB_DIR)}")


if __name__ == "__main__":
    ingest()
