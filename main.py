"""
main.py — Simple CLI for the College Chatbot.

This is the entry point for asking questions. It assumes you have
already run `python ingestion.py` to index your college documents.

Usage:
    python main.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from rag import answer_question


def main():
    print("=" * 60)
    print("  College Chatbot (RAG)")
    print("=" * 60)
    print()
    print("Ask questions about your college documents.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        # Prompt the user for a question
        question = input("Enter your question:\n> ").strip()

        # Check for exit commands
        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        # Skip empty input
        if not question:
            print("Please enter a question.\n")
            continue

        print("\nSearching documents and generating answer...\n")

        # Run the full RAG pipeline
        result = answer_question(question)

        # Display the answer
        print("-" * 40)
        print("Answer:")
        print(result["answer"])

        # Display source references
        if result["sources"]:
            print("\nSources:")
            for src in result["sources"]:
                print(f"  • {src['source']}, page {src['page']}")
        else:
            print("\nSources: No relevant documents found.")

        print(f"\n(Used {result['chunks_used']} document chunks)")
        print("-" * 40)
        print()


if __name__ == "__main__":
    main()
