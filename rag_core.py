import os
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Please set your GROQ_API_KEY in the .env file")

PERSIST_DIR = "chroma_store"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant answering questions about a document.
Use ONLY the provided context to answer.

If the user gives just a topic (e.g. "CNN"), explain that topic using
whatever the context says about it.
Answer from the context even if it only covers part of the question.
Only if the context has nothing relevant at all, say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}
Question:
{question}
"""
        )
    ]
)


def get_vectorstore():
    """Open the persistent Chroma store (shared by all uploaded PDFs)."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )


def get_retriever(vectorstore, source_name=None):
    """MMR retriever. If source_name is given, search only that PDF."""
    search_kwargs = {"k": 6, "fetch_k": 30, "lambda_mult": 0.7}
    if source_name:
        search_kwargs["filter"] = {"source_name": source_name}
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )


def _ids_for(vectorstore, source_name):
    return vectorstore.get(where={"source_name": source_name})["ids"]


def add_pdf(vectorstore, pdf_path, display_name=None):
    """Add a PDF to the store. Re-uploading the same name replaces it."""
    name = display_name or os.path.basename(pdf_path)

    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    # Drop empty chunks and table-of-contents / index noise (dot leaders)
    chunks = [
        c for c in chunks
        if c.page_content.strip()
        and not re.search(r"(\.\s?){10,}", c.page_content)
    ]

    if not chunks:
        raise ValueError(
            f"No text found in '{name}'. It may be a scanned PDF (needs OCR)."
        )

    for chunk in chunks:
        chunk.metadata["source_name"] = name

    # Avoid duplicates if the same file is uploaded again
    old_ids = _ids_for(vectorstore, name)
    if old_ids:
        vectorstore.delete(ids=old_ids)

    vectorstore.add_documents(chunks)
    return len(chunks)


def list_documents(vectorstore):
    """Return [{'name': ..., 'chunks': ...}, ...] for every loaded PDF."""
    data = vectorstore.get(include=["metadatas"])
    counts = Counter(
        (m or {}).get("source_name", "unknown") for m in data["metadatas"]
    )
    return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]


def delete_document(vectorstore, source_name):
    """Delete all chunks of one PDF. Returns number of chunks removed."""
    ids = _ids_for(vectorstore, source_name)
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def answer_question(vectorstore, query, source_name=None):
    """Return (answer_text, sources). Optionally restrict to one PDF."""
    retriever = get_retriever(vectorstore, source_name)
    docs = retriever.invoke(query)

    if not docs:
        return "No documents found. Please upload a PDF first.", []

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    response = llm.invoke(final_prompt)

    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        sources.append({
            "text": doc.page_content,
            "page": (page + 1) if isinstance(page, int) else None,
            "file": doc.metadata.get("source_name")
        })

    return response.content, sources