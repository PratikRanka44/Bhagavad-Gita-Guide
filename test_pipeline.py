"""
Quick terminal test of the full pipeline, without needing Streamlit.

Usage:
    export GROQ_API_KEY=gsk_...
    python test_pipeline.py
"""

from rag.pipeline import GitaGuidePipeline


def main():
    pipeline = GitaGuidePipeline()

    print("=== Gita Guide - CLI test ===")
    situation = input("Describe your situation: ").strip()
    language = input("Output language (English/Hindi/Marathi) [English]: ").strip() or "English"

    result = pipeline.run(situation, output_language=language)

    if not result["in_scope"]:
        print("\n--- Out of scope ---")
        print(result["response"])
        return

    print("\n--- Extracted themes ---")
    print(result["theme_summary"])

    print("\n--- Top candidate verses (retrieval) ---")
    for c in result["candidates"]:
        print(f"  [{c['score']:.3f}] Ch{c['chapter']}.{c['verse']} - {c['themes']}")

    v = result["selected_verse"]
    print(f"\n--- Selected verse: Chapter {v['chapter']}, Verse {v['verse']} ---")
    print(f"Reasoning: {v['reasoning']}")
    print(f"Sanskrit: {v['sanskrit']}")

    print("\n--- Response ---")
    print(result["response"])


if __name__ == "__main__":
    main()