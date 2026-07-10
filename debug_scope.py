"""
debug_scope.py
--------------
One-off diagnostic: shows exactly why a given input was accepted or
rejected by the pipeline's two scope gates (LLM classifier + similarity
threshold), so we can tell which one is misfiring.

Usage:
    python debug_scope.py "I'm scared to leave my stable job and start my own business"
"""

import sys

from rag import agents
from rag.retriever import VerseRetriever
from rag.pipeline import MIN_RELEVANCE_SCORE

situation = sys.argv[1] if len(sys.argv) > 1 else "I'm scared to leave my stable job and start my own business"

print(f"Situation: {situation}\n")

scope = agents.analyze_situation(situation)
print(f"[Gate 1] LLM classifier -> in_scope={scope['in_scope']}, themes='{scope['themes']}'")

retriever = VerseRetriever()
search_query = f"{situation} {scope['themes']}"
candidates = retriever.search(search_query, top_k=5)

print(f"\n[Gate 2] Retrieval scores (threshold = {MIN_RELEVANCE_SCORE}):")
for c in candidates:
    flag = "OK" if c["score"] >= MIN_RELEVANCE_SCORE else "BELOW THRESHOLD"
    print(f"  [{c['score']:.3f}] {flag}  Ch{c['chapter']}.{c['verse']} - {c['themes']}")

top_score = candidates[0]["score"] if candidates else 0
print(f"\nTop score: {top_score:.3f} vs threshold {MIN_RELEVANCE_SCORE}")
if not scope["in_scope"]:
    print("=> REJECTED at Gate 1 (LLM classifier said out of scope)")
elif top_score < MIN_RELEVANCE_SCORE:
    print("=> REJECTED at Gate 2 (top retrieval score below MIN_RELEVANCE_SCORE)")
else:
    print("=> Would PASS both gates")