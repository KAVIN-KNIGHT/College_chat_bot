"""
app.py — Streamlit Chat UI for the College RAG Chatbot.

A modern, chatbot-style interface that connects to the RAG pipeline.
Run with:  streamlit run app.py
"""

import streamlit as st
import time

# ─── Page Config (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="CampusAI - College Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─── Cache heavy resources so they load only ONCE ───────────────────
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """Load the RAG pipeline once and cache it across all reruns."""
    from rag import answer_question
    return answer_question


answer_question = load_rag_pipeline()


# ─── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 5rem;
    max-width: 780px;
}

/* ── Hero Banner ── */
.hero {
    text-align: center;
    padding: 1.8rem 1rem 1rem;
    margin-bottom: 0.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(102, 126, 234, 0.15);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(102,126,234,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(118,75,162,0.08) 0%, transparent 50%);
    pointer-events: none;
}
.hero-icon {
    font-size: 2.4rem;
    margin-bottom: 0.3rem;
}
.hero h1 {
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #a78bfa, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 0;
    font-weight: 400;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.7rem;
    color: #34d399;
    font-weight: 500;
    margin-top: 0.8rem;
}
.status-pill .dot {
    width: 6px;
    height: 6px;
    background: #34d399;
    border-radius: 50%;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Chat bubbles ── */
.stChatMessage {
    border-radius: 14px !important;
    padding: 0.6rem 0.8rem !important;
}

/* ── Source tags ── */
.src-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(102, 126, 234, 0.08);
    border: 1px solid rgba(102, 126, 234, 0.18);
    border-radius: 8px;
    padding: 3px 10px;
    margin: 3px 4px 3px 0;
    font-size: 0.72rem;
    color: #818cf8;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.15), transparent);
    margin: 0.6rem 0 0.8rem;
    border: none;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] li, [data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🎓</div>
    <h1>CampusAI</h1>
    <p>Ask anything about your college — powered by RAG</p>
    <div class="status-pill"><span class="dot"></span> Knowledge base connected</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hey there! I'm **CampusAI** 🎓\n\nI can help you with questions about admissions, courses, policies, facilities, campus life, and more. What would you like to know?",
            "sources": [],
        }
    ]

# ─── Render Chat History ─────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            tags = "".join(
                f'<span class="src-tag">📄 {s["source"]}, p.{s["page"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(f"<div style='margin-top:6px'>{tags}</div>", unsafe_allow_html=True)

# ─── Chat Input ──────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your college..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching documents..."):
            result = answer_question(prompt)

        st.markdown(result["answer"])

        if result["sources"]:
            tags = "".join(
                f'<span class="src-tag">📄 {s["source"]}, p.{s["page"]}</span>'
                for s in result["sources"]
            )
            st.markdown(f"<div style='margin-top:6px'>{tags}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 CampusAI")
    st.markdown("Your intelligent college assistant.")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown(
        "1. Your question gets embedded into a vector\n"
        "2. ChromaDB finds the most relevant document chunks\n"
        "3. Context + question are sent to Gemini\n"
        "4. You get an accurate, sourced answer"
    )
    st.markdown("---")

    st.markdown("### Settings")
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hey there! I'm **CampusAI** 🎓\n\nI can help you with questions about admissions, courses, policies, facilities, campus life, and more. What would you like to know?",
                "sources": [],
            }
        ]
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit + ChromaDB + Gemini")
