"""
retrieval.py — Retrieve relevant document chunks from ChromaDB.

Given a user question, this module:
1. Converts the question into an embedding vector.
2. Queries ChromaDB for the top-K most similar chunks.
3. Returns the chunk texts along with their metadata (source, page, etc.).
"""

import chromadb

from config import CHROMA_DB_DIR, COLLECTION_NAME, DISTANCE_METRIC, TOP_K
from embeddings import get_embedding


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most relevant document chunks for a given question using Cosine Similarity.

    Args:
        question: The user's natural-language question.
        top_k:    How many chunks to retrieve (default from config).

    Returns:
        A list of dicts, each containing:
            {
                "text":              "chunk content...",
                "source":            "academic_regulations.pdf",
                "page":              12,
                "chunk_index":       3,
                "distance":          0.18,  # Cosine distance (1 - cosine_similarity)
                "similarity":        0.82,  # Cosine similarity score (higher = more similar)
                "cosine_similarity": 0.82
            }
        Sorted by relevance (highest similarity / lowest distance first).
    """
    # Connect to the persistent ChromaDB with cosine similarity space
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )

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
        dist = results["distances"][0][i]
        # In ChromaDB cosine space: distance = 1 - cosine_similarity
        # Hence: cosine_similarity = 1 - distance
        sim = round(1.0 - dist, 4) if dist is not None else None
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": dist,
            "similarity": sim,
            "cosine_similarity": sim,
        })

    return chunks
