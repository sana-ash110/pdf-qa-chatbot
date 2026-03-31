"""
app.py
------
Streamlit UI for the PDF QA chatbot.

Run with:
    streamlit run app.py

Project structure expected:
    pdf-qa-bot/
    ├── app.py
    ├── .env
    ├── requirements.txt
    └── utils/
        ├── pdf_loader.py
        ├── chunking.py
        ├── embeddings.py
        ├── vector_store.py
        └── qa_chain.py
"""

import os
import io
import sys
import tempfile

# ── Make utils/ importable ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import streamlit as st
from dotenv import load_dotenv

from pdf_loader   import load_pdf
from chunking     import chunk_pages
from vector_store import build_vector_store
from qa_chain     import QAChain

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "PDF QA Bot",
    page_icon  = "📄",
    layout     = "wide",
)

# ── Session state defaults ────────────────────────────────────────────────────

if "store"         not in st.session_state: st.session_state.store         = None
if "chain"         not in st.session_state: st.session_state.chain         = None
if "messages"      not in st.session_state: st.session_state.messages      = []
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = None
if "pdf_processed" not in st.session_state: st.session_state.pdf_processed = False
if "page_count"    not in st.session_state: st.session_state.page_count    = 0
if "chunk_count"   not in st.session_state: st.session_state.chunk_count   = 0


# ── Helper functions — defined BEFORE they are used ──────────────────────────

def _render_sources(sources: list[dict], images: list[dict]) -> None:
    """Renders source citations and image results below an answer."""
    if not sources and not images:
        return

    with st.expander("Sources & context", expanded=False):

        if sources:
            st.caption("**Text sources retrieved:**")
            for s in sources:
                page_label = f"Page {s['page_number']}" if s["page_number"] else "Unknown page"
                score_pct  = int(s["score"] * 100)
                st.markdown(
                    f"**{page_label}** &nbsp; `{score_pct}% match` &nbsp; *{s['source']}*"
                )
                st.markdown(
                    f"> {s['excerpt']}{'…' if len(s['excerpt']) == 200 else ''}"
                )

        if images:
            st.caption("**Relevant images found:**")
            cols = st.columns(min(len(images), 3))
            for col, img in zip(cols, images):
                if img.get("image_data"):
                    col.image(
                        io.BytesIO(img["image_data"]),
                        caption          = f"Page {img['page_number']} · {int(img['score'] * 100)}% match",
                        use_column_width = True,
                    )


@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes: bytes, filename: str, use_clip: bool):
    """
    Loads, chunks and indexes a PDF.
    Cached by Streamlit — re-uploading the same file skips re-embedding.
    Returns: (store, page_count, chunk_count)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pages  = load_pdf(tmp_path, extract_images=True)
        chunks = chunk_pages(pages, chunk_size=500, overlap=100)
        store  = build_vector_store(chunks, pages, use_clip_for_text=use_clip)
    finally:
        os.unlink(tmp_path)

    return store, len(pages), len(chunks)


def reset_chat():
    """Clears conversation when a new PDF is uploaded."""
    st.session_state.messages = []
    if st.session_state.chain:
        st.session_state.chain.clear_history()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 PDF QA Bot")
    st.caption("Upload a PDF and ask questions about it.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type = ["pdf"],
        help = "Supports any text-based PDF.",
    )

    st.divider()
    st.subheader("Settings")

    model = st.selectbox(
        "Groq model",
        options = [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index = 0,
    )

    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value = 1,
        max_value = 10,
        value     = 5,
        help      = "More chunks = more context, but slower.",
    )

    min_score = st.slider(
        "Min similarity score",
        min_value = 0.0,
        max_value = 1.0,
        value     = 0.3,
        step      = 0.05,
        help      = "Raise to filter out weakly-matched chunks.",
    )

    use_clip = st.toggle(
        "Use CLIP for text",
        value = True,
        help  = (
            "ON: CLIP encodes text + images in the same space "
            "(enables image retrieval). OFF: faster MiniLM, text only."
        ),
    )

    st.divider()

    if st.session_state.pdf_processed:
        st.success(f"**{st.session_state.pdf_name}**")
        col1, col2 = st.columns(2)
        col1.metric("Pages",  st.session_state.page_count)
        col2.metric("Chunks", st.session_state.chunk_count)

        if st.button("Clear chat", use_container_width=True):
            reset_chat()
            st.rerun()


# ── PDF processing ────────────────────────────────────────────────────────────

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename   = uploaded_file.name

    # New PDF uploaded — reset everything
    if filename != st.session_state.pdf_name:
        reset_chat()
        st.session_state.pdf_processed = False
        st.session_state.pdf_name      = filename
        st.session_state.store         = None
        st.session_state.chain         = None

    if not st.session_state.pdf_processed:
        with st.spinner(f"Processing **{filename}**… this may take a minute on first run."):
            try:
                store, page_count, chunk_count = process_pdf(
                    file_bytes, filename, use_clip
                )
                st.session_state.store         = store
                st.session_state.chain         = QAChain(
                    store,
                    model     = model,
                    top_k     = top_k,
                    min_score = min_score,
                )
                st.session_state.page_count    = page_count
                st.session_state.chunk_count   = chunk_count
                st.session_state.pdf_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
                st.stop()

    # Keep chain settings in sync with sidebar
    if st.session_state.chain:
        st.session_state.chain.set_top_k(top_k)
        st.session_state.chain.set_model(model)
        st.session_state.chain.min_score = min_score


# ── Landing page ──────────────────────────────────────────────────────────────

if not st.session_state.pdf_processed:
    st.markdown("## 👈 Upload a PDF to get started")
    st.markdown(
        "Ask questions about any PDF. Uses **CLIP** for retrieval "
        "and **Groq** (llama-3.3-70b) to generate answers with page citations."
    )
    st.stop()


# ── Chat history (re-rendered on every Streamlit rerun) ───────────────────────
# _render_sources is defined above so it exists by the time this loop runs.

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-render sources/images that were stored with assistant messages
        if msg["role"] == "assistant":
            _render_sources(
                msg.get("sources", []),
                msg.get("images",  []),
            )


# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about the PDF…"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.chain.ask(prompt)
                answer = result["answer"]
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                result = {"answer": answer, "sources": [], "images": []}

        st.markdown(answer)
        _render_sources(result["sources"], result["images"])

    # Persist message + metadata for re-rendering on next rerun
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "images":  result.get("images",  []),
    })