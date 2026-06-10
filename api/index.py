from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import json
import faiss
from app.rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize RAG Engine
engine = RAGEngine()

# Load pre-built index and chunks
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.json"

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    engine.index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        engine.chunks = json.load(f)
else:
    # Build on the fly if needed (fallback)
    KB_PATH = "data/knowledge_base.txt"
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            engine.create_index(f.read())

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = engine.query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pathlib import Path

PUBLIC_DIR = Path(__file__).parent.parent / "public"

@app.get("/")
async def read_index():
    return FileResponse(PUBLIC_DIR / "index.html")

app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")