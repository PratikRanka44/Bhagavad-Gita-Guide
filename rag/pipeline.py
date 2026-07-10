"""
pipeline.py
-----------
Orchestrates the full flow:

user situation -> analyze_situation (scope check + themes) -> retrieve
candidates -> similarity-threshold safety check -> select_verse
-> compose_response (in chosen output language)

If the input isn't a genuine personal situation (math, trivia, small talk,
etc.), the pipeline short-circuits and returns a polite "not applicable"
result instead of forcing an unrelated verse onto it.
"""

from typing import Dict

from .retriever import VerseRetriever
from . import agents

# Below this cosine-similarity score, even the "best" retrieved verse isn't
# a real match — acts as a loose backup guard in case the LLM scope-classifier
# mis-labels something as in_scope. Kept deliberately low: real testing showed
# genuinely valid situations scoring as low as ~0.31-0.37 with this embedding
# model, so a stricter cutoff risks rejecting legitimate input over minor
# wording differences (a single typo was enough to flip a valid match below
# 0.35). Gate 1 (the LLM classifier) does the actual semantic judgment; this
# is just a sanity net for genuinely nonsensical retrieval.
MIN_RELEVANCE_SCORE = 0.15

_OUT_OF_SCOPE_MESSAGE = {
    "English": (
        "That doesn't seem to be a personal situation the Gita would speak to directly — "
        "this tool is meant for things like decisions, worries, conflicts, or emotional "
        "struggles. Try describing something you're actually facing."
    ),
    "Hindi": (
        "यह कोई व्यक्तिगत परिस्थिति प्रतीत नहीं होती जिस पर गीता सीधे मार्गदर्शन दे सके — "
        "यह उपकरण निर्णय, चिंता, संघर्ष या भावनात्मक कठिनाइयों जैसी बातों के लिए है। "
        "कृपया अपनी वास्तविक स्थिति बताएं।"
    ),
}


class GitaGuidePipeline:
    def __init__(self):
        self.retriever = VerseRetriever()

    def run(self, user_input: str, output_language: str = "English", top_k: int = 5) -> Dict:
        """
        Returns a dict with:
          - in_scope: bool
          - theme_summary: extracted themes used for retrieval (if in_scope)
          - candidates: the raw top_k retrieved verses (if in_scope)
          - selected_verse: the verse chosen, with 'reasoning' (if in_scope)
          - response: final text — either composed guidance or the
            out-of-scope message
        """
        scope = agents.analyze_situation(user_input)

        if not scope["in_scope"]:
            lang_key = "Hindi" if output_language.lower().startswith("hi") else "English"
            return {
                "in_scope": False,
                "theme_summary": None,
                "candidates": [],
                "selected_verse": None,
                "response": _OUT_OF_SCOPE_MESSAGE[lang_key],
            }

        theme_summary = scope["themes"]

        # Retrieve using both the raw input and the extracted theme summary,
        # concatenated, so we capture literal detail plus distilled theme.
        search_query = f"{user_input} {theme_summary}"
        candidates = self.retriever.search(search_query, top_k=top_k)

        # Backup guard: if even the top match is a weak fit, treat as
        # out-of-scope rather than forcing the closest-of-a-bad-lot verse.
        if not candidates or candidates[0]["score"] < MIN_RELEVANCE_SCORE:
            lang_key = "Hindi" if output_language.lower().startswith("hi") else "English"
            return {
                "in_scope": False,
                "theme_summary": theme_summary,
                "candidates": candidates,
                "selected_verse": None,
                "response": _OUT_OF_SCOPE_MESSAGE[lang_key],
            }

        selected_verse = agents.select_verse(user_input, candidates)
        response_text = agents.compose_response(user_input, selected_verse, output_language)

        return {
            "in_scope": True,
            "theme_summary": theme_summary,
            "candidates": candidates,
            "selected_verse": selected_verse,
            "response": response_text,
        }