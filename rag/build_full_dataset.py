"""
build_full_dataset.py
----------------------
Converts the open-source vedicscriptures/bhagavad-gita dataset
(https://github.com/vedicscriptures/bhagavad-gita, GPLv3) into this
project's schema, producing data/gita_verses.json with all verses.

Source fields used per verse:
  - slok              -> sanskrit
  - transliteration    -> transliteration
  - tej.ht (Swami Tejomayananda's Hindi translation)   -> hindi
  - siva.et (Swami Sivananda's English translation)    -> english
  - speaker            -> speaker (who is speaking: Krishna, Arjuna, etc.)

"themes" is left as an empty list — see generate_themes.py to optionally
auto-tag themes via an LLM (requires your own GROQ_API_KEY), which
significantly improves retrieval quality over relying on raw translation
text alone.

Usage:
    git clone https://github.com/vedicscriptures/bhagavad-gita.git
    python scripts/build_full_dataset.py --source ./bhagavad-gita --out data/gita_verses.json
"""

import argparse
import glob
import json
import os
import re


def clean_sanskrit(raw: str) -> str:
    text = raw.strip()
    # Drop trailing verse-number marker, e.g. ||२-४७|| or ||2-47||
    text = re.sub(r"\|\|[^|]*\|\|\s*$", "", text).strip()
    text = text.replace("\n", " ")
    text = text.replace("||", "॥").replace("|", "।")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in "।॥":
        text += "॥"
    return text


def clean_transliteration(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\|\|[^|]*\|\|\s*$", "", text).strip()
    text = text.replace("\n", " ")
    # pada-separator periods have surrounding whitespace -> comma
    text = re.sub(r"\s+\.\s+", ", ", text)
    text = re.sub(r"\s+\.$", "", text)
    # remaining mid-word periods are avagraha substitutes -> apostrophe
    text = text.replace(".", "'")
    text = text.replace("||", ", ").replace("|", ", ")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(", ").strip()
    return text


def clean_hindi(raw: str) -> str:
    text = raw.strip()
    # Strip leading "।।2.47।।" style verse-number marker
    text = re.sub(r"^।।\s*[\d०-९.\-,\s]+।।\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_english(raw: str) -> str:
    text = raw.strip()
    # Strip leading "2.47 " style verse-number marker
    text = re.sub(r"^[\d]+\.[\d]+(?:-[\d]+)?\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build(source_dir: str, out_path: str) -> None:
    slok_files = sorted(glob.glob(os.path.join(source_dir, "slok", "*.json")))
    if not slok_files:
        raise SystemExit(
            f"No verse files found under {source_dir}/slok — check --source path."
        )

    verses = []
    for fp in slok_files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)

        verses.append(
            {
                "chapter": d["chapter"],
                "verse": d["verse"],
                "speaker": d.get("speaker", ""),
                "sanskrit": clean_sanskrit(d["slok"]),
                "transliteration": clean_transliteration(d["transliteration"]),
                "hindi": clean_hindi(d["tej"]["ht"]),
                "english": clean_english(d["siva"]["et"]),
                "themes": [],  # populate via generate_themes.py (optional)
            }
        )

    verses.sort(key=lambda v: (v["chapter"], v["verse"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(verses)} verses to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="Path to cloned vedicscriptures/bhagavad-gita repo"
    )
    parser.add_argument("--out", default="data/gita_verses.json")
    args = parser.parse_args()
    build(args.source, args.out)