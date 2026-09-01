"""
config.py — Central configuration for the RAG pipeline.

All tunable parameters live here so you can adjust the pipeline
without digging through multiple files.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────
# Directory where you place your college PDF documents.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Directory where ChromaDB stores its persistent database.
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ─── ChromaDB ────────────────────────────────────────────────────────
# Name of the ChromaDB collection that holds all document chunks.
COLLECTION_NAME = "college_knowledge"

# ─── Embedding Model ────────────────────────────────────────────────
# ChromaDB's built-in DefaultEmbeddingFunction uses all-MiniLM-L6-v2
# via ONNX runtime. No separate configuration is needed — if you want
# a different model, swap the embedding function in embeddings.py.

# ─── Chunking ───────────────────────────────────────────────────────
# Maximum number of characters per chunk.
CHUNK_SIZE = 500

# Number of overlapping characters between consecutive chunks.
# Overlap helps preserve context that falls on chunk boundaries.
CHUNK_OVERLAP = 100

# ─── Retrieval ──────────────────────────────────────────────────────
# Number of most-relevant chunks to retrieve for each question.
TOP_K = 5
