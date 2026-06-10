import os
import json
import faiss
import numpy as np
from app.rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

def build():
    # No API key needed for local embeddings
    engine = RAGEngine()
    
    with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
        kb_text = f.read()

    print("Building index using SentenceTransformers...")
    engine.create_index(kb_text)
    
    # Save chunks
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(engine.chunks, f)
    
    # Save FAISS index
    faiss.write_index(engine.index, "data/index.faiss")
    print("DONE: Index and chunks saved to data/ directory.")

if __name__ == "__main__":
    build()
