"""
generate_themes.py
-------------------
Optional preprocessing step: uses an LLM to tag each verse in
data/gita_verses.json with short theme phrases (e.g. "grief", "fear of
failure"), which meaningfully improves retrieval quality over matching
raw translation text alone.

This is a separate, one-time (or occasional) batch job — not something
the live app calls per-request — since tagging ~700+ verses means ~700+
API calls. It's resumable: verses that already have non-empty "themes"
are skipped, so you can re-run safely if it's interrupted partway through.

Usage:
    export GROQ_API_KEY=gsk_...
    python scripts/generate_themes.py

Takes roughly 10-20 minutes for the full corpus depending on Groq's
current rate limits for your account tier.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gita_verses.json")
MODEL = os.environ.get("GITA_GUIDE_MODEL", "openai/gpt-oss-120b")

client = Groq()

SYSTEM_PROMPT = (
    "You tag a Bhagavad Gita verse with short theme phrases describing what kind "
    "of real-life situation it speaks to (e.g. 'grief', 'fear of failure', "
    "'career confusion', 'anger', 'seeking inner peace', 'leadership', "
    "'self-doubt'). Respond with ONLY a comma-separated list of 3-6 short theme "
    "phrases, no preamble, no explanation."
)


def tag_verse(verse: dict) -> list:
    user = f"Verse (Chapter {verse['chapter']}.{verse['verse']}): {verse['english']}"
    kwargs = dict(
        model=MODEL,
        max_tokens=150,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    if MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"

    response = client.chat.completions.create(**kwargs)
    content = (response.choices[0].message.content or "").strip()

    if not content:
        kwargs["max_tokens"] = 400
        response = client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()

    themes = [t.strip() for t in content.split(",") if t.strip()]
    return themes


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        verses = json.load(f)

    todo = [v for v in verses if not v.get("themes")]
    print(f"{len(verses)} verses total, {len(todo)} need tagging.")

    for i, verse in enumerate(todo):
        try:
            verse["themes"] = tag_verse(verse)
            print(f"[{i+1}/{len(todo)}] BG {verse['chapter']}.{verse['verse']} -> {verse['themes']}")
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] BG {verse['chapter']}.{verse['verse']} FAILED: {e}")
            continue

        # Save progress every 20 verses so a crash/rate-limit doesn't lose work.
        if (i + 1) % 20 == 0:
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(verses, f, ensure_ascii=False, indent=2)
            print(f"  -- progress saved ({i+1}/{len(todo)}) --")

        time.sleep(0.3)  # gentle pacing to stay well under rate limits

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    print("Done. All themes saved to", DATA_PATH)


if __name__ == "__main__":
    main()