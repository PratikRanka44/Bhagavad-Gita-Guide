"""
retriever.py
------------
Loads the verse dataset, builds embeddings for each verse (using its English
translation + themes, since that's the layer users will semantically match
against), and exposes a `search(query, top_k)` method.

Uses sentence-transformers with a multilingual model so that Hindi *and*
English situation inputs both retrieve well, even though the indexed text
is in English.
"""

import json
import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gita_verses.json")

# Multilingual model: works across Hindi, English (and Sanskrit reasonably),
# good balance of quality vs size for local/full-resource use.
#
# Deployment note: this model is ~1GB in memory, which can exceed Streamlit
# Community Cloud's 1GB RAM limit on its own. For cloud deployment, override
# with the lighter alternative via an env var (no code change needed):
#   GITA_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
# That model is ~470MB and still handles Hindi/English well, just with
# somewhat less nuanced matching than the larger mpnet model.
MODEL_NAME = os.environ.get("GITA_EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")


class VerseRetriever:
    def __init__(self, data_path: str = DATA_PATH, model_name: str = MODEL_NAME):
        with open(data_path, "r", encoding="utf-8") as f:
            self.verses: List[Dict] = json.load(f)

        self.model = SentenceTransformer(model_name)

        # Build the searchable text for each verse: translation + themes (if any).
        # This is deliberately NOT the raw Sanskrit — retrieval works far
        # better when matched against natural-language meaning + tags.
        self._corpus_texts = []
        for v in self.verses:
            text = v["english"]
            if v.get("themes"):
                text += " Themes: " + ", ".join(v["themes"])
            self._corpus_texts.append(text)
        self._embeddings = self.model.encode(
            self._corpus_texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top_k verse records most relevant to the query, each with
        a similarity score attached."""
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self._embeddings @ query_vec  # cosine similarity (vectors are normalized)
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            verse = dict(self.verses[idx])
            verse["score"] = float(scores[idx])
            results.append(verse)
        return results


if __name__ == "__main__":
    # Quick manual smoke test
    retriever = VerseRetriever()
    test_query = "I'm scared to quit my stable job and start my own business"
    results = retriever.search(test_query, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] Ch{r['chapter']}.{r['verse']} - {r['themes']}")
