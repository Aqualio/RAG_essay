# Research Paper Analyst (Local LLM + RAG + CrewAI)

An AI-powered **research paper analysis app** built with **Streamlit**, **CrewAI**, and **local open-source LLMs**.  
Upload a PDF research paper, get an **automatic structured summary**, and **ask questions with citation-based answers** using Retrieval-Augmented Generation (RAG).

Runs **fully locally** (no OpenAI key required) using:

- `microsoft/phi-2` for generation  
- `all-MiniLM-L6-v2` for embeddings  
- `Chroma` or `FAISS` for vector search  

---

## Features

- Upload any research paper (PDF)  
- Automatic structured summary:
  - Title  
  - Abstract  
  - Methodology  
  - Key Findings  
  - Contributions  
  - Limitations  
  - Future Work  
- Semantic search over the paper (RAG)  
- Multi-agent pipeline with **CrewAI**
  - Retriever agent → finds relevant passages  
  - Generator agent → produces citation-based answers  
- Local embeddings + local LLM (privacy-friendly)  
- Chat interface with history  
- Execution trace viewer (debug/learning mode)  
- Automatic ChromaDB cleanup  

---

## Architecture

PDF → Text Extraction → Chunking → Embeddings → Vector DB
↓
Similarity Search
↓
Retriever Agent (CrewAI) → Relevant Passages
↓
Generator Agent (CrewAI) → Answer with Citations


---

## Tech Stack

- **UI**: Streamlit  
- **Agents**: CrewAI  
- **LLM (local)**: HuggingFace `microsoft/phi-2`  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector DB**: Chroma (fallback to FAISS)  
- **PDF parsing**: PyPDF2  
- **RAG utilities**: LangChain  

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/research-paper-analyst.git
cd research-paper-analyst
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install requirement.txt
```

## Run the App
```bash
streamlit run app.py
```

## How to Use

1. Upload a PDF research paper from the sidebar
2. Click “Process Document”
3. Wait for:
  - Structured summary generation
  - Vector index creation
4. Ask questions in the chat box
5. View:
  - AI answer with citations
  - Execution trace (optional)

## Configuration
### Change LLM (optional)
Inside load_local_llm():

```python
model="microsoft/phi-2"
```
You can swap for:
 - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (faster)
 - `mistralai/Mistral-7B-Instruct-v0.2` (better quality, needs GPU)

### Use GPU

```python
device = 0
```

## Project Structure
 
```code
.
├── app.py                # Streamlit application
├── chroma_db/            # Local vector database (auto-created)
├── README.md
└── requirements.txt
```

## Agents (CrewAI)
### Retriever Agent
  - Goal: Find the most relevant passages
  - Input: User question
  - Output: Top-k semantic chunks
### Generator Agent
  - Goal: Produce a citation-based answer
  - Constraint: Must use only retrieved passages
  - Output: Structured answer with [1], [2], …

## Limitations
- Local LLM quality < GPT-4-level models
- Long papers are truncated during summarization (first ~6000 chars)
- CPU inference can be slow
- PDF text extraction quality depends on formatting

## Future Improvements
- Multi-PDF support
- Hybrid search (BM25 + embeddings)
- Better citation grounding (chunk IDs → page numbers)
- Table & figure extraction
- Streaming token output
- GPU auto-detection
- Persistent vector DB per document

## License
MIT License 















```bash

```
