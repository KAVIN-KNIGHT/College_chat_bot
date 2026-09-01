"""
retrieval.py — Retrieve relevant document chunks from ChromaDB.

Given a user question, this module:
1. Converts the question into an embedding vector.
2. Queries ChromaDB for the top-K most similar chunks.
3. Returns the chunk texts along with their metadata (source, page, etc.).
"""

import chromadb

from config import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K
from embeddings import get_embedding


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most relevant document chunks for a given question.

    Args:
        question: The user's natural-language question.
        top_k:    How many chunks to retrieve (default from config).

    Returns:
        A list of dicts, each containing:
            {
                "text":        "chunk content...",
                "source":      "academic_regulations.pdf",
                "page":        12,
                "chunk_index": 3,
                "distance":    0.42   # lower = more similar
            }
        Sorted by relevance (most relevant first).
    """
    # Connect to the persistent ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Check if the collection has any documents
    if collection.count() == 0:
        print("[retrieval] Warning: ChromaDB collection is empty. Run ingestion.py first.")
        return []

    # Embed the user's question using the same model as ingestion
    question_embedding = get_embedding(question)

    # Query ChromaDB for the top-K most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack the results into a clean list of dicts
    # ChromaDB returns lists-of-lists because you can query multiple
    # questions at once; we only have one, so we take index [0].
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
        })

    return chunks
