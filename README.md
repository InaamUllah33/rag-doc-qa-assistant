# Doc-QA RAG Assistant

Upload any PDF and ask questions about it. The assistant answers using
**only** the document's content (Retrieval-Augmented Generation), with
page-number citations — no hallucinated answers from general knowledge.

## How it works (architecture)

```
Upload PDF
    │
    ▼
Extract text (PyPDFLoader)
    │
    ▼
Split into ~800-char chunks with overlap (RecursiveCharacterTextSplitter)
    │
    ▼
Embed each chunk into a vector (OpenAI text-embedding-3-small)
    │
    ▼
Store vectors in FAISS index (in-memory, keyed by session_id)

--- on question ---

User question
    │
    ▼
Embed the question the same way
    │
    ▼
FAISS similarity search -> top 4 most relevant chunks
    │
    ▼
Stuff chunks + question into a prompt -> GPT-4o-mini
    │
    ▼
Answer grounded in the retrieved chunks, with source page numbers
```

**Why RAG instead of just asking the LLM directly?** LLMs don't know
your private/specific document, and they can hallucinate. RAG grounds
the answer in retrieved, real text from the document itself, and you
can show exactly which page the answer came from.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs` for interactive Swagger UI.

## API

- `POST /upload` — multipart file upload (PDF) -> returns `session_id`
- `POST /ask` — `{"session_id": "...", "question": "..."}` -> answer + sources
- `DELETE /session/{id}` — clear a document from memory

## Roadmap (next steps for this project)

- [ ] Swap FAISS (in-memory) for Qdrant (persistent, production-grade)
- [ ] Add streaming responses (token-by-token, like ChatGPT)
- [ ] Add a simple frontend (or just demo via Swagger UI for now)
- [ ] Docker + deployment
- [ ] Add conversation memory (follow-up questions)
