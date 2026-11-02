import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"
API_ASK_URL = f"{API_BASE}/ask"
API_UPLOAD_URL = f"{API_BASE}/upload"
API_DOCS_URL = f"{API_BASE}/documents"

st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖", layout="centered")


def fetch_documents():
    try:
        res = requests.get(API_DOCS_URL, timeout=10)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except requests.exceptions.RequestException:
        pass
    return None  # backend unreachable


def show_sources(sources):
    with st.expander("🔍 View Source Documents"):
        for i, source in enumerate(sources):
            label = f"**Source Chunk {i + 1}**"
            if source.get("file"):
                label += f" — {source['file']}"
            if source.get("page"):
                label += f", page {source['page']}"
            st.markdown(label)
            st.text(source.get("text", ""))
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files and st.button("Process Documents", use_container_width=True):
        with st.spinner("Processing and embedding documents..."):
            for f in uploaded_files:
                files = {"file": (f.name, f.getvalue(), "application/pdf")}
                try:
                    res = requests.post(API_UPLOAD_URL, files=files)
                    if res.status_code == 200:
                        st.success(f"Loaded '{f.name}'")
                    else:
                        st.error(f"{f.name}: {res.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend.")
                    break
        st.session_state.uploader_key += 1  # reset the uploader
        st.rerun()

    st.divider()
    st.subheader("Loaded documents")

    documents = fetch_documents()
    if documents is None:
        st.error("Backend not reachable. Run: `uvicorn backend:app --reload`")
        documents = []
    elif not documents:
        st.caption("No documents loaded yet.")

    for doc in documents:
        c1, c2 = st.columns([4, 1])
        c1.write(f"📄 {doc['name']}")
        c1.caption(f"{doc['chunks']} chunks")
        if c2.button("🗑️", key=f"del_{doc['name']}", help=f"Delete {doc['name']}"):
            requests.delete(API_DOCS_URL, params={"name": doc["name"]})
            st.rerun()

    st.divider()
    options = ["All documents"] + [d["name"] for d in documents]
    selected = st.selectbox("Search in", options)
    selected_doc = None if selected == "All documents" else selected

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main chat ----------------
st.title("📚 Interactive RAG Assistant")
st.write("Ask questions across all your PDFs, or pick a single one in the sidebar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_ASK_URL,
                    json={"question": prompt, "document": selected_doc},
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    answer = f"Error from server: {error_detail}"
                    sources = []
            except requests.exceptions.ConnectionError:
                answer = (
                    "Could not connect to the FastAPI backend. "
                    "Make sure it is running: `uvicorn backend:app --reload`"
                )
                sources = []

            st.markdown(answer)
            if sources:
                show_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })