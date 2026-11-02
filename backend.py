import os
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from rag_core import (
    get_vectorstore,
    add_pdf,
    list_documents,
    delete_document,
    answer_question,
)

app = FastAPI(title="RAG API", description="FastAPI backend for multi-PDF Q&A")

print("Initializing RAG Pipeline Components...")
vectorstore = get_vectorstore()
print("RAG system created and ready!")


class QueryRequest(BaseModel):
    question: str
    document: Optional[str] = None  # None = search all PDFs


class Source(BaseModel):
    text: str
    page: Optional[int] = None
    file: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        chunks = add_pdf(vectorstore, tmp_path, display_name=file.filename)
        return {"message": f"Successfully processed {file.filename}", "chunks": chunks}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/documents")
def get_documents():
    try:
        return {"documents": list_documents(vectorstore)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
def remove_document(name: str):
    try:
        removed = delete_document(vectorstore, name)
        if removed == 0:
            raise HTTPException(status_code=404, detail=f"'{name}' not found.")
        return {"message": f"Deleted '{name}'", "chunks_removed": removed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    query = request.question
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        answer, sources = answer_question(vectorstore, query, request.document)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))