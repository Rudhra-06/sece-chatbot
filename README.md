# SECE RAG Chatbot

A complete, production-ready Retrieval-Augmented Generation (RAG) chatbot for **Sri Eshwar College of Engineering (SECE)**. It is built with a FastAPI backend and a premium, modern HTML/CSS frontend, optimized for Vercel deployment.

## Features
- **RAG Architecture**: Chunks the knowledge base, generates vector embeddings, stores them in a FAISS index, and retrieves context using Cosine Similarity.
- **Fast and Smart LLM**: Generates answers using Groq's high-speed `llama-3.3-70b-versatile` model.
- **Hybrid Embeddings**: Employs Hugging Face's `all-MiniLM-L6-v2` online API for high-quality text embeddings. Includes an automatic offline fallback for sandbox/local environments without internet access.
- **Premium User Interface**: Includes a responsive chatbot interface with a sidebar containing quick links and a modal showing retrieval scores and context chunks for each answer.
- **Vercel Deployable**: Pre-configured with a `vercel.json` routing configuration and minimal dependencies to fit within Vercel's serverless size limits.

## Project Structure
```text
├── api/
│   └── index.py            # FastAPI entry point for Vercel
├── app/
│   └── rag_engine.py       # Core RAG logic (Chunking, Embedding, FAISS, LLM)
├── data/
│   ├── knowledge_base.txt  # Core SECE knowledge base text
│   ├── index.faiss         # Pre-built FAISS index
│   └── chunks.json         # Text chunks mapped to index
├── public/
│   ├── index.html          # Web UI layout
│   ├── styles.css          # Premium stylesheet
│   └── script.js           # UI interaction logic
├── scripts/
│   ├── build_index.py      # Script to pre-build the FAISS index
│   └── test_rag.py         # Testing script for CLI questions
├── .env.example            # Environment template
└── vercel.json             # Vercel deployment configuration
```

## Getting Started

### 1. Local Setup
Clone the repository, open the directory, and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Optional: Add `HF_TOKEN=your_huggingface_token` if you have one to bypass Hugging Face API rate limits, though it works without one.)*

### 3. Build the Index
Run the index builder to generate your FAISS vector database locally:
```bash
python scripts/build_index.py
```

### 4. Start the Application
Run the local development server:
```bash
uvicorn api.index:app --reload
```
Open your browser and navigate to `http://localhost:8000`.

---

## Testing CLI Questions
To test the pipeline via the console, run:
```bash
python scripts/test_rag.py
```

## Deploying to Vercel
This app is ready to deploy on Vercel:
1. Initialize a Vercel project:
   ```bash
   vercel
   ```
2. Add your `GROQ_API_KEY` to the Vercel project environment variables in the dashboard.
3. Deploy to production:
   ```bash
   vercel --prod
   ```
