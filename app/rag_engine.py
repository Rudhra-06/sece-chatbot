import os
import numpy as np
import requests
from groq import Groq
import faiss
from typing import List, Dict

class RAGEngine:
    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"
        self.hf_api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.index = None
        self.chunks = []
        self.using_fallback = False

    def chunk_text(self, text: str, chunk_size: int = 150, overlap: int = 20) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def get_fallback_embeddings(self, texts: List[str], dimension: int = 384) -> List[List[float]]:
        # Deterministic offline fallback using simple word-hashing
        self.using_fallback = True
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vector = np.zeros(dimension, dtype=np.float32)
            if words:
                for w in words:
                    # Simple hash to distribute words across dimensions
                    idx = hash(w) % dimension
                    vector[idx] += 1.0
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            embeddings.append(vector.tolist())
        return embeddings

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        try:
            response = requests.post(self.hf_api_url, headers=headers, json={"inputs": texts}, timeout=10)
            if response.status_code == 200:
                self.using_fallback = False
                return response.json()
            elif response.status_code == 503:
                # Model loading, wait and retry once
                import time
                time.sleep(5)
                response = requests.post(self.hf_api_url, headers=headers, json={"inputs": texts}, timeout=10)
                if response.status_code == 200:
                    self.using_fallback = False
                    return response.json()
        except Exception as e:
            print(f"Warning: Embedding API failed ({e}). Falling back to local offline embeddings.")
            
        return self.get_fallback_embeddings(texts)

    def create_index(self, text: str):
        self.chunks = self.chunk_text(text)
        embeddings = self.get_embeddings(self.chunks)
        embedding_matrix = np.array(embeddings).astype('float32')
        
        faiss.normalize_L2(embedding_matrix)
        dimension = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embedding_matrix)
        
    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        # If we created the index using fallback, query must also use fallback
        if self.using_fallback:
            query_embedding = np.array(self.get_fallback_embeddings([query])).astype('float32')
        else:
            query_embedding = np.array(self.get_embeddings([query])).astype('float32')
            
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "text": self.chunks[idx],
                    "score": float(score)
                })
        return results

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        context_text = "\n\n".join([c['text'] for c in context_chunks])
        
        system_prompt = (
            "You are a helpful chatbot for Sri Eshwar College of Engineering (SECE). "
            "Use ONLY the provided context to answer the user's question. "
            "If the answer is not in the context, refer the user to sece.ac.in. "
            "Keep the answer concise and professional."
        )
        
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception as e:
            # Fallback response if Groq API fails (e.g. no internet locally)
            return (
                f"[Offline Mode] Based on the context: {context_text[:200]}... "
                f"For full answers, please visit sece.ac.in. (Error: {e})"
            )

    def query(self, user_query: str) -> Dict:
        relevant_chunks = self.retrieve(user_query)
        answer = self.generate_answer(user_query, relevant_chunks)
        return {
            "answer": answer,
            "retrieval_results": relevant_chunks,
            "offline": self.using_fallback
        }
