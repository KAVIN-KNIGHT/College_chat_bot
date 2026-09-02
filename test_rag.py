"""Test: verify LLM call works, then test the full RAG flow."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from llm import generate_answer

# -- Test 1: Raw LLM call --
print("=" * 50)
print("TEST 1: Direct LLM call")
print("=" * 50)
try:
    response = generate_answer(
        context="CampusAI University was founded in 2005.",
        question="When was CampusAI University founded?"
    )
    print(f"Response: {response}")
    print("[PASS] LLM call works!\n")
except Exception as e:
    print(f"[FAIL] LLM call failed: {e}\n")

# -- Test 2: Full RAG flow (user query -> retrieval -> LLM -> output) --
print("=" * 50)
print("TEST 2: Full RAG pipeline")
print("=" * 50)
from rag import answer_question

question = "What are the library timings?"
print(f"User Query: {question}\n")

result = answer_question(question)

print("Answer:")
print(result["answer"])
print()
print("Sources:")
for src in result["sources"]:
    print(f"  - {src['source']}, page {src['page']}")
print(f"\n({result['chunks_used']} chunks used)")
