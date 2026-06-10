import os
import json
from app.rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

def test():
    # RAGEngine will pick up GROQ_API_KEY from .env
    engine = RAGEngine()
    
    # Load data for testing
    if not os.path.exists("data/knowledge_base.txt"):
        print("Error: knowledge_base.txt not found.")
        return
        
    with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
        kb_text = f.read()
    
    print("Initializing engine for testing (Groq + SentenceTransformers)...")
    engine.create_index(kb_text)
    
    sample_questions = [
        "What is the TNEA code for SECE?",
        "Which departments are available in SECE?",
        "What are the research facilities at SECE?",
        "How can I get admission for 2025?",
        "Who are the top recruiters for placements?"
    ]
    
    print("\n--- Running Sample Questions on Groq ---\n")
    for q in sample_questions:
        print(f"Q: {q}")
        try:
            result = engine.query(q)
            print(f"A: {result['answer']}")
            print("Top Retrieval Score:", result['retrieval_results'][0]['score'])
        except Exception as e:
            print(f"Error during query: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test()
