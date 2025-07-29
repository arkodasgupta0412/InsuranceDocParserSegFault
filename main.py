from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import List
import requests
from dotenv import load_dotenv
import os

from utils.chunker import chunk_documents
from utils.loader import load_document_from_bytes
from utils.embedder import get_vectorstore
from utils.llm_response import generate_structured_answers

load_dotenv()

app = FastAPI(
    title="Insurance Parser API",
    version="1.0",
    root_path="/api/v1"
)

BEARER_TOKEN = os.getenv("BEARER_TOKEN")

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

@app.get("/")
def greet():
    return {"message": "Insurance Parser API is live at /api/v1."}

@app.post("/hackrx/run")
def process_doc(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must start with 'Bearer '")
    if authorization.split(" ")[1] != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        response = requests.get(request.documents, timeout=60)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    content_type = response.headers.get("Content-Type", "").lower()
    url_lower = str(request.documents).lower()

    if 'pdf' in url_lower or 'application/pdf' in content_type:
        file_type = "PDF"
    elif 'docx' in url_lower or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
        file_type = "DOCX"
    elif 'eml' in url_lower or 'message/rfc822' in content_type:
        file_type = "EML"
    elif 'msg' in url_lower or 'application/vnd.ms-outlook' in content_type:
        file_type = "MSG"
    else:
        file_type = "TEXT"

    docs = load_document_from_bytes(response.content, file_type)
    chunks = chunk_documents(docs)
    vectordb = get_vectorstore(chunks)
    answers = generate_structured_answers(request.questions, vectordb)

    return {
        "answers": answers
    }

