"""
agents.py
---------
Three LLM-driven stages of the pipeline:

1. analyze_situation  -> extract the core theme/emotion from raw user input
2. select_verse        -> given retrieved candidates, pick the single best verse
3. compose_response    -> generate the final situational explanation in the
                          user's chosen output language, grounded strictly in
                          the selected verse's actual text (never regenerated)

Uses the Groq Python SDK (OpenAI-compatible chat completions). Set
GROQ_API_KEY as an environment variable before running.
"""

import json
import os
from typing import Dict, List

from groq import Groq

# Groq's model lineup changes fairly often (models get deprecated with a
# migration notice). As of writing, openai/gpt-oss-120b is the recommended
# general-purpose model. Check https://console.groq.com/docs/models for the
# current list, or override via the GITA_GUIDE_MODEL env var.
MODEL = os.environ.get("GITA_GUIDE_MODEL", "openai/gpt-oss-120b")

client = Groq()  # reads GROQ_API_KEY from env automatically


def _call(system: str, user: str, max_tokens: int = 600, temperature: float = 0.7) -> str:
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    # openai/gpt-oss-* models on Groq are reasoning models: they spend part of
    # max_tokens on internal chain-of-thought before writing the final answer.
    # With low max_tokens, reasoning can eat the whole budget and leave the
    # actual response empty. Forcing low reasoning effort keeps more of the
    # budget available for the answer itself.
    if MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"

    response = client.chat.completions.create(**kwargs)
    content = (response.choices[0].message.content or "").strip()

    if not content:
        # Retry once with a larger budget — covers the case where reasoning
        # alone consumed all of max_tokens on the first attempt.
        kwargs["max_tokens"] = max(max_tokens * 3, 500)
        response = client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()

    return content


def analyze_situation(user_input: str) -> Dict:
    """Classify whether this is something the Gita can meaningfully speak to —
    either a personal situation or a genuine philosophical/reflective question —
    and if so extract its core theme(s) for retrieval.

    Returns: {"in_scope": bool, "themes": str}
    "in_scope" is False for things like math questions, trivia, random factual
    queries, small talk, or anything with no reflective, philosophical, or
    emotional weight — so the pipeline can skip forcing an irrelevant verse
    onto it.
    """
    system = (
        "You screen inputs for a Bhagavad Gita guidance tool. Mark in_scope=true for "
        "TWO kinds of input:\n"
        "1. A real personal situation, decision, emotional struggle, or ethical dilemma "
        "(e.g. 'I'm scared to leave my job', 'my friend betrayed me').\n"
        "2. A genuine philosophical, psychological, or reflective question about human "
        "life, nature, or meaning — even if phrased abstractly or in third person, not "
        "as a personal confession (e.g. 'why do humans resist change', 'what is the "
        "meaning of happiness', 'why do people fear death', 'what makes someone "
        "virtuous'). The Gita is as much a philosophical text as an emotional one, so "
        "these count.\n\n"
        "Mark in_scope=false for: math/arithmetic, trivia, factual/general-knowledge "
        "lookups (capitals, dates, sports scores), coding/technical requests, small "
        "talk, greetings, or anything with no reflective, philosophical, or emotional "
        "substance at all.\n\n"
        "Respond ONLY with valid JSON, no other text:\n"
        '{"in_scope": true or false, "themes": "comma-separated short theme phrases, '
        'empty string if not in_scope"}'
    )
    raw = _call(system, user_input, max_tokens=150, temperature=0.1)
    raw_clean = raw.strip().strip("`")
    if raw_clean.lower().startswith("json"):
        raw_clean = raw_clean[4:].strip()

    try:
        parsed = json.loads(raw_clean)
        return {
            "in_scope": bool(parsed.get("in_scope", False)),
            "themes": parsed.get("themes", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        # If the classifier response is malformed, default to in_scope=True
        # rather than blocking a legitimate situation over a parsing hiccup —
        # the downstream similarity threshold in the retriever is the backup.
        return {"in_scope": True, "themes": user_input}


def select_verse(user_input: str, candidates: List[Dict]) -> Dict:
    """Given retrieved candidate verses, ask the model to pick the single best fit.
    Returns the chosen verse dict (unmodified, straight from the dataset) plus
    a short 'reasoning' field explaining the choice."""

    def _format_candidate(i, c):
        line = f"{i+1}. Chapter {c['chapter']}, Verse {c['verse']} - \"{c['english']}\""
        if c.get("themes"):
            line += f" (themes: {', '.join(c['themes'])})"
        return line

    candidate_summary = "\n".join(_format_candidate(i, c) for i, c in enumerate(candidates))

    system = (
        "You are selecting the single most relevant Bhagavad Gita verse for a person's "
        "real-life situation, from a shortlist of candidates. You must pick ONLY from the "
        "numbered list given — never invent a verse or number not in the list. "
        "Respond ONLY with valid JSON: {\"choice_number\": <int>, \"reasoning\": \"<one sentence>\"}"
    )
    user = f"Situation: {user_input}\n\nCandidates:\n{candidate_summary}"

    raw = _call(system, user, max_tokens=350, temperature=0.1)
    raw_clean = raw.strip().strip("`")
    if raw_clean.startswith("json"):
        raw_clean = raw_clean[4:].strip()

    try:
        parsed = json.loads(raw_clean)
        choice_idx = int(parsed["choice_number"]) - 1
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        # Fallback: if parsing fails, trust the retriever's top result rather
        # than letting a malformed LLM response break the pipeline.
        choice_idx = 0
        reasoning = "Selected by retrieval similarity (verse-selection parse fallback)."

    choice_idx = max(0, min(choice_idx, len(candidates) - 1))
    chosen = dict(candidates[choice_idx])
    chosen["reasoning"] = reasoning
    return chosen


def compose_response(user_input: str, verse: Dict, output_language: str) -> str:
    """Generate the final situational guidance, grounded in the selected verse.
    output_language should be 'English', 'Hindi', or 'Marathi'.

    Note: there's no dedicated Marathi field in the verse dataset — Marathi
    responses are grounded in the Hindi translation instead (both are
    Devanagari-script, Sanskrit-derived languages, so this grounds the LLM's
    Marathi output more naturally than grounding off English would). This is
    fine because the app never displays a raw translation field to the user
    directly — only Sanskrit and transliteration are shown verbatim; the
    translation fields are just LLM grounding context either way.
    """
    lang = output_language.lower()
    if lang.startswith("hi") or lang.startswith("mar"):
        translation = verse["hindi"]
    else:
        translation = verse["english"]

    system = (
        f"You explain Bhagavad Gita wisdom in response to what someone has shared — either "
        f"a personal situation or a genuine reflective/philosophical question — in {output_language}. "
        f"Write your ENTIRE response in {output_language} only.\n\n"
        "STRICT LENGTH LIMIT: 4-6 sentences total, no more. No headers, no numbered "
        "sections, no bullet lists. Write it as a short, warm paragraph (or two at most) "
        "someone could read in 15 seconds.\n\n"
        "Cover, in that short space:\n"
        "- what the verse means, in one plain sentence\n"
        "- how it directly answers or illuminates what they asked — a concrete link to "
        "their situation if they shared one, or a clear answer if they asked a reflective "
        "question\n"
        "- optionally, one small, specific takeaway — an action for a situation, or a "
        "grounding insight for a philosophical question — not a list of five\n\n"
        "Do not invent any Sanskrit or claim verse numbers other than the one given below. "
        "Be direct and specific to what they said, not generic."
    )

    user = (
        f"Person's input: {user_input}\n\n"
        f"Verse to ground your answer in — Chapter {verse['chapter']}, Verse {verse['verse']}:\n"
        f"Sanskrit: {verse['sanskrit']}\n"
        f"Translation: {translation}\n"
        f"Why this verse was chosen: {verse.get('reasoning', '')}"
    )

    return _call(system, user, max_tokens=500)