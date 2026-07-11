"""
Streamlit UI for the Gita Guide project.

Run with:
    export GROQ_API_KEY=gsk_...
    streamlit run app.py
"""

import base64
import os
import random

import streamlit as st

from rag.pipeline import GitaGuidePipeline

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "assets", "background.mp3")

st.set_page_config(page_title="Gita Guide", page_icon="🕉️", layout="centered")

# ---------------------------------------------------------------------------
# Theme: dusk-over-the-battlefield indigo, saffron/gold, warm parchment.
# Base app colors live in .streamlit/config.toml — this CSS layers in fonts,
# the manuscript-style verse card, decorative dividers, and button styling
# that Streamlit's theme tokens alone don't cover.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Noto+Serif+Devanagari:wght@500;600&family=Inter:wght@400;500&display=swap');

    /* Hide the "Press Ctrl+Enter to submit form" hint below text_area.
       Streamlit doesn't expose a parameter for this. */
    div[data-testid="stTextAreaInstructions"],
    div[data-testid="InputInstructions"] {
        visibility: hidden;
        height: 0;
        margin: 0;
        padding: 0;
    }

    /* Subtle dusk-sky vignette instead of a flat background */
    .stApp {
        background: radial-gradient(ellipse at top, #1c2740 0%, #141B2E 55%, #0f1522 100%);
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        letter-spacing: 0.02em;
    }

    /* Manuscript-style card for the retrieved verse */
    .verse-card {
        background: linear-gradient(180deg, #F7EEDD 0%, #F0E4CC 100%);
        border: 1px solid #D4A537;
        border-radius: 10px;
        padding: 1.6rem 1.8rem;
        margin: 0.5rem 0 1.2rem 0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .verse-card .verse-heading {
        font-family: 'Cormorant Garamond', Georgia, serif;
        color: #7A2E22;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }
    .verse-card .verse-sanskrit {
        font-family: 'Noto Serif Devanagari', 'Cormorant Garamond', serif;
        color: #5A2A12;
        font-size: 1.35rem;
        font-weight: 600;
        line-height: 1.8;
        margin: 0 0 0.5rem 0;
    }
    .verse-card .verse-translit {
        color: #7A6142;
        font-style: italic;
        font-size: 0.95rem;
    }

    /* Decorative divider: a small centered flourish instead of a plain rule */
    .om-divider {
        text-align: center;
        color: #D4A537;
        font-size: 1.1rem;
        margin: 1.4rem 0;
        opacity: 0.8;
        letter-spacing: 0.6rem;
    }

    /* Primary buttons: saffron-to-maroon gradient pill */
    button[kind="primary"] {
        background: linear-gradient(135deg, #D4A537 0%, #B5451F 100%) !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        color: #1B0F08 !important;
    }
    button[kind="primary"]:hover {
        filter: brightness(1.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Banner renders immediately (cheap), before the slower model load below ---
st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 0.6rem 0;">
        <div style="font-size:3rem; line-height:1;">🕉️</div>
        <h1 style="margin:0.3rem 0 0.2rem 0;">Gita Guide</h1>
        <p style="color:#B8AE99; font-size:0.95rem; max-width:480px; margin:0 auto;">
            Describe a situation you're facing. This tool retrieves a genuinely relevant
            verse from the Bhagavad Gita and explains how it applies — grounded in a
            curated, verified verse dataset (not model-generated Sanskrit).
        </p>
    </div>
    <div class="om-divider">🪷 ✦ 🪷</div>
    """,
    unsafe_allow_html=True,
)


# --- Background music, single-click toggle ---
# Browsers won't autoplay audio-with-sound on page load without a prior user
# gesture — that's a deliberate browser policy, not fixable in code. Once the
# user clicks ANYTHING on the page (including this button itself), the
# browser treats the page as "activated" and allows audio to start.
if "music_enabled" not in st.session_state:
    st.session_state.music_enabled = False


def _toggle_music():
    # Runs BEFORE the script reruns, so state is already correct by the
    # time widgets are laid out again — avoids the "one click behind" bug
    # that happens when a button's own label depends on the state it sets.
    st.session_state.music_enabled = not st.session_state.music_enabled


button_label = "🔇 Stop background music" if st.session_state.music_enabled else "🔊 Enable background music"
st.button(button_label, on_click=_toggle_music, key="music_toggle_btn")

if st.session_state.music_enabled:
    if os.path.exists(AUDIO_PATH):
        with open(AUDIO_PATH, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <audio autoplay loop>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "⚠️ No background.mp3 found in assets/ — see assets/README.md "
            "for where to get a free one."
        )


GITA_FACTS = [
    "The Bhagavad Gita is a dialogue between Prince Arjuna and Krishna, set on a battlefield.",
    "The text records a deep philosophical conversation between Prince Arjuna and Lord Krishna on a battlefield.",
    "The Gita is part of the Mahabharata, an ancient Indian epic.",
    "The Gita is a 700-verse Hindu scripture that is part of the Indian epic Mahabharata.",
    "\"Gita\" means \"song\" — Bhagavad Gita translates roughly to \"Song of God.\"",
]


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return GitaGuidePipeline()


loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown(
        f"""
        <div style="text-align:center; padding:2.5rem 0;">
            <div style="font-size:3rem; display:inline-block;">🕉️</div>
            <p style="color:#B8AE99; font-size:0.9rem; max-width:420px; margin:1rem auto 0;">
                {random.choice(GITA_FACTS)}
            </p>
        </div>
        <style>
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    pipeline = load_pipeline()
loading_placeholder.empty()

with st.form("situation_form"):
    output_language = st.radio("Response language", ["English", "Hindi", "Marathi"], horizontal=True)

    user_input = st.text_area(
        "Your situation",
        placeholder="e.g. I'm anxious about a big decision and scared I'll regret it either way.",
        height=120,
    )

    submitted = st.form_submit_button("Get guidance", type="primary")

if submitted:
    if not user_input.strip():
        st.warning("Please describe your situation first.")
    else:
        with st.spinner("🦚 Reflecting on your situation 🦚"):
            result = pipeline.run(user_input, output_language=output_language)

        if not result["in_scope"]:
            st.info(result["response"])
        else:
            verse = result["selected_verse"]

            st.markdown(
                f"""
                <div class="verse-card">
                    <div class="verse-heading">Bhagavad Gita {verse['chapter']}.{verse['verse']}</div>
                    <div class="verse-sanskrit">{verse['sanskrit']}</div>
                    <div class="verse-translit">{verse['transliteration']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(result["response"])

           