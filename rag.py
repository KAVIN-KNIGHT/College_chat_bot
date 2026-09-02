"""
rag.py — RAG (Retrieval Augmented Generation) orchestrator.

This module ties everything together:
1. Takes a user question.
2. Retrieves relevant document chunks from ChromaDB.
3. Builds a context string from the chunks.
4. Calls the existing generate_answer() function.
5. Returns the answer along with source references.
"""

from retrieval import retrieve
from llm import generate_answer


def format_context(chunks: list[dict]) -> str:
    """
    Combine retrieved chunks into a single context string.

    Each chunk is labelled with its source and page number so the LLM
    can attribute information back to specific documents.
    """
    if not chunks:
        return "No relevant documents were found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source: {chunk['source']}, Page {chunk['page']}]"
        context_parts.append(f"--- Chunk {i} {header} ---\n{chunk['text']}")

    return "\n\n".join(context_parts)


def format_sources(chunks: list[dict]) -> list[dict]:
    """
    Extract unique source references from the retrieved chunks.

    Returns a deduplicated list of {source, page} dicts, preserving
    the order in which they first appeared.
    """
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
            })
    return sources


def answer_question(question: str, top_k: int = None) -> dict:
    """
    Full RAG pipeline: question → retrieve (cosine similarity) → prompt → LLM → answer.

    Args:
        question: The user's natural-language question.
        top_k:    Override the default number of chunks to retrieve.

    Returns:
        {
            "answer":      "The minimum attendance requirement is...",
            "sources":     [
                {"source": "academic_regulations.pdf", "page": 12},
                ...
            ],
            "chunks_used": 5,
            "chunks":      [list of retrieved chunk dicts with similarity scores]
        }
    """
    # Step 1: Retrieve relevant chunks from ChromaDB using Cosine Similarity
    kwargs = {}
    if top_k is not None:
        kwargs["top_k"] = top_k

    chunks = retrieve(question, **kwargs)

    # Step 2: Build the context string from retrieved chunks
    context = format_context(chunks)

    # Step 3: Call the existing LLM function
    answer = generate_answer(context=context, question=question)

    # Step 4: Gather source references
    sources = format_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "chunks": chunks,
    }
