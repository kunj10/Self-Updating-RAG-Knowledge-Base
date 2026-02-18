import os
import numpy as np
import faiss
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_db = None
embed_model = None
groq_client = None

KNOWLEDGE_CHUNKS = [
    "MyTax is a modern GST compliance and billing software designed for Indian businesses.",
    "MyTax supports automated GST return filing.",
    "MyTax supports e-invoicing.",
    "MyTax allows multi-user access for enterprise teams.",
    "The MyTax support team can be reached at support@mytax.com."
]

# --- Resilient Agentic Helper Functions ---

def is_retryable_error(exception):
    """Checks if the error is a 429 (Rate Limit) or 5xx (Server Error)."""
    status = getattr(exception, "status_code", None)
    if status and status in (429, 500, 503):
        return True
    return "429" in str(exception) or "500" in str(exception) or "503" in str(exception)

# This decorator will handle the 429 Rate Limit errors automatically
@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_groq_with_retry(prompt):
    """Calls Groq with a built-in retry mechanism for rate limits."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# --- FastAPI Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, embed_model, groq_client

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        groq_client = Groq(api_key=api_key)

        # Load HuggingFace embedding model locally (22MB, zero cost)
        logger.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        logger.info("Building FAISS Vector Database...")
        embeddings = embed_model.encode(KNOWLEDGE_CHUNKS, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        dimension_size = embeddings.shape[1]
        vector_db = faiss.IndexFlatL2(dimension_size)
        vector_db.add(embeddings)
        logger.info(f"Successfully loaded {vector_db.ntotal} documents into the Vector DB.")

    yield

app = FastAPI(title="MyTax AutoRAG API", lifespan=lifespan)

# --- Schemas ---

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: str

# --- Endpoints ---

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not groq_client or not vector_db or not embed_model:
        raise HTTPException(status_code=500, detail="Database or Client not initialized")

    try:
        # 1. Embed the user's question locally
        question_vector = embed_model.encode([request.query], normalize_embeddings=True)
        question_vector = np.array(question_vector, dtype=np.float32)

        # 2. Retrieve Top 2 chunks
        distances, indices = vector_db.search(question_vector, k=2)
        retrieved_chunks = [KNOWLEDGE_CHUNKS[i] for i in indices[0]]
        retrieved_context = "\n".join(retrieved_chunks)

        # 3. Step 1: Initial Answer Generation
        initial_prompt = f"""
        You are a helpful customer support AI for MyTax.
        Answer the user's question based ONLY on the provided context. 
        If you don't know the answer based on the context, say "I don't know."
        
        Context:
        {retrieved_context}
        
        Question:
        {request.query}
        """

        initial_answer = call_groq_with_retry(initial_prompt)

        # 4. Step 2: Agentic Critique & Hallucination Check
        critique_prompt = f"""
        As an AI Auditor, evaluate if this answer is strictly grounded in the provided context.
        Context: {retrieved_context}
        Answer: {initial_answer}

        If the answer contains hallucinations or info not in the context, rewrite it to be 100% faithful.
        Otherwise, return the original answer exactly.
        """

        final_answer = call_groq_with_retry(critique_prompt)

        return QueryResponse(answer=final_answer, context=retrieved_context)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))