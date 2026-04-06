# ResearchRAG — AI Research Assistant

A Retrieval-Augmented Generation (RAG) pipeline that ingests research papers (PDFs) and answers natural language queries with grounded, citation-backed responses.

## Stack
- **LangChain** — orchestration and chain management
- **FAISS** — local vector store for fast similarity search
- **Groq API** — LLM inference (LLaMA 3 8B)
- **HuggingFace Embeddings** — sentence-transformers for document embedding

## How It Works
1. PDFs are loaded and split into chunks
2. Chunks are embedded and stored in a FAISS vector index
3. On query, top-k relevant chunks are retrieved
4. Retrieved context + query is passed to Groq-hosted LLaMA 3
5. Answer is returned with source citations

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Wik-Hrit/ResearchRAG.git
cd ResearchRAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Add research papers
Place PDF files inside the `papers/` folder.

### 5. Run
```bash
python app.py
```

## Usage
```
ResearchRAG ready. Type your question (or 'exit' to quit).

You: What are the limitations of BPTT in long sequences?
--- Answer ---
According to the paper on reservoir computing...
--- Sources ---
  papers/reservoir_computing.pdf | Page 4
```

## Project Structure
```
ResearchRAG/
├── app.py               # Main pipeline
├── requirements.txt
├── .env                 # Your API key (not committed)
├── papers/              # Drop your PDFs here
└── faiss_index/         # Auto-generated vector index
```
