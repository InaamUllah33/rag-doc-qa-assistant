"""
RAG (Retrieval-Augmented Generation) core logic.

Flow: PDF upload -> extract text -> split into chunks -> embed chunks ->
store in FAISS vector DB -> on question, retrieve relevant chunks ->
pass chunks + question to LLM -> get grounded answer.

100% free mode: both embeddings and the LLM run via Google's Gemini
API free tier -> generous limits (100 requests/day on Gemini 2.5 Pro),
no local model download, no torch. Only needs GOOGLE_API_KEY
(free from https://aistudio.google.com/apikey).
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# In-memory store: maps a session_id -> FAISS vectorstore for that doc.
# In production this would be a persistent vector DB (Qdrant/Pinecone),
# but FAISS in-memory is perfect for learning + a local demo.
_vector_stores: dict[str, FAISS] = {}


def _get_embeddings():
    """
    Embeddings computed via Google's Gemini embedding model (remote
    API call, free tier). No local download, no torch.
    """
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def _get_llm():
    """
    Gemini 2.5 Flash: fast + free-tier friendly, good default for a
    RAG QA task. (Use gemini-2.5-pro if you want higher quality and
    don't mind the lower free daily request limit.)
    """
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.9)


def process_document(file_path: str, session_id: str) -> int:
    """
    Load a PDF, split it into overlapping chunks, embed each chunk,
    and store the embeddings in a FAISS index keyed by session_id.

    Returns the number of chunks created.
    """
    # 1. Load PDF -> list of Document objects (usually one per page)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Split into smaller chunks.
    #    Why chunk? LLMs have limited context, and smaller chunks give
    #    more precise retrieval (you don't pull in a whole 10-page doc
    #    just to answer a one-line question).
    #    chunk_overlap keeps some continuity between chunks so we don't
    #    cut a sentence/idea in half and lose meaning.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    # 3. Embed each chunk into a vector (a list of floats that captures
    #    semantic meaning) and store in FAISS for fast similarity search.
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    _vector_stores[session_id] = vector_store
    return len(chunks)


def answer_question(question: str, session_id: str) -> dict:
    """
    Given a question, retrieve the most relevant chunks for this
    session's document and ask the LLM to answer using only that context.
    """
    if session_id not in _vector_stores:
        raise ValueError("No document found for this session. Upload a PDF first.")

    vector_store = _vector_stores[session_id]

    # k=4 -> pull the top 4 most similar chunks to the question.
    # This is the "R" (Retrieval) in RAG.
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # --- DEBUG: manually see what the retriever actually pulled ---
    retrieved_docs = retriever.invoke(question)
    print("\n===== RETRIEVED CHUNKS =====")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {i+1} (page {doc.metadata.get('page', '?')}) ---")
        print(doc.page_content[:300])
    print("===== END CHUNKS =====\n")
    # --- END DEBUG ---

    llm = _get_llm()

    # RetrievalQA wires it together: retrieve chunks -> stuff them into
    # a prompt -> LLM generates an answer grounded in those chunks.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa_chain.invoke({"query": question})

    sources = [
        {"page": doc.metadata.get("page", "?"), "snippet": doc.page_content[:150]}
        for doc in result["source_documents"]
    ]

    return {"answer": result["result"], "sources": sources}


def clear_session(session_id: str) -> None:
    _vector_stores.pop(session_id, None)