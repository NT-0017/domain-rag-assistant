# app.py
# PURPOSE: A simple chat web page for the RAG assistant (built with Streamlit).
import streamlit as st
import os
import pandas as pd

from generator import assistant, collection, MODEL_NAME, GeminiError
from store import add_pdf

st.set_page_config(page_title="Domain RAG Assistant", page_icon="📚")
st.title("📚 Domain RAG Assistant")
st.caption("Ask questions about your PDFs. Answers are generated from your "
           "documents and cite their sources.")

# ---- Sidebar: list the documents in the database ----
with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Add PDF files", type=["pdf"],
                                accept_multiple_files=True)
    if uploaded:
        os.makedirs("data", exist_ok=True)
        if "processed" not in st.session_state:
            st.session_state.processed = set()
        for f in uploaded:
            key = (f.name, f.size)
            if key in st.session_state.processed:
                continue
            path = os.path.join("data", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            with st.spinner(f"Reading {f.name}..."):
                result = add_pdf(collection, path)
            st.session_state.processed.add(key)
            if result["status"] == "added":
                st.success(f"{f.name}: added {result['chunks']} chunks.")
                if result["skipped_pages"]:
                    st.warning(f"{result['skipped_pages']} scanned page(s) were skipped "
                               "(OCR is off).")
            elif result["status"] == "exists":
                st.info(f"{f.name} is already in the database.")
            else:
                st.error(f"{f.name}: no readable text found. It may be a scanned PDF "
                         "(OCR is off).")

    st.header("Documents")
    metas = collection.get(include=["metadatas"])["metadatas"]
    names = sorted({m["file"] for m in metas})
    rows = []
    for name in names:
        file_metas = [m for m in metas if m["file"] == name]
        rows.append({
            "Document": name,
            "Pages": str(len({m["page"] for m in file_metas})),
            "Chunks": str(len(file_metas)),
        })
    st.table(pd.DataFrame(rows).set_index("Document"))
    selected = st.selectbox("Search in", ["All documents"] + names)
    file_filter = None if selected == "All documents" else selected

    st.write(f"{collection.count()} chunks stored")
    st.caption(f"Model: {MODEL_NAME}")

    chat = st.session_state.get("messages", [])
    if chat:
        chat_text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in chat)
        st.download_button("Download chat", chat_text, file_name="chat.txt")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Chat history lives in st.session_state (it survives reruns) ----
if "messages" not in st.session_state:
    st.session_state.messages = []

def show_sources(sources):
    """Show the sources in a collapsed box, grouped by file."""
    if not sources:
        return
    by_file = {}
    for f, p in sources:
        by_file.setdefault(f, []).append(p)
    n = len(by_file)
    with st.expander(f"📄 Sources ({n} document{'s' if n > 1 else ''})"):
        for f, pages in by_file.items():
            pages = sorted(set(pages))
            label = "page" if len(pages) == 1 else "pages"
            st.markdown(f"**{f}**: {label} {', '.join(str(p) for p in pages)}")

def show_message(role, content, sources=None):
    with st.chat_message(role):
        st.markdown(content)
        show_sources(sources)


for msg in st.session_state.messages:
    show_message(msg["role"], msg["content"], msg.get("sources"))
# ---- Welcome screen (only while the chat is empty) ----
if not st.session_state.messages and "pending" not in st.session_state:
    st.markdown(
        "**What you can ask** (use your own topics):\n"
        "- A question: *What is ...?*\n"
        "- A page summary: *Summarize page 4*, or *Explain page 4 as if I am 5*\n"
        "- Study notes: *Generate notes for ...*\n"
        "- Practice questions: *Generate a True False exercise on ...*"
    )
    if st.button("Try it: summarize page 1", key="example_page1"):
        st.session_state.pending = "Summarize page 1"
        st.rerun()
# ---- New question ----
question = st.chat_input("Ask a question about your PDFs")
if not question:
    question = st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    show_message("user", question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = assistant(question, file=file_filter,
                                            history=st.session_state.messages[:-1])
            except GeminiError as e:
                answer, sources = f"⚠️ {e}",[]
            except Exception as e: 
                answer, sources = f"Sorry, something unexpected went wrong: {e}", []

        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()