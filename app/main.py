import os
import numpy as np
import faiss
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_db = None
genai_client = None

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
    if isinstance(exception, errors.ClientError):
        # Retry on Rate Limits (429) or Server Errors (500, 503)
        return "429" in str(exception) or "500" in str(exception) or "503" in str(exception)
    return False

# This decorator will handle the 429 Resource Exhausted errors automatically
@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_gemini_with_retry(model_name, prompt):
    """Calls Gemini with a built-in retry mechanism for rate limits."""
    return genai_client.models.generate_content(
        model=model_name,
        contents=prompt
    )

# --- FastAPI Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, genai_client
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai_client = genai.Client(api_key=api_key)
        
        logger.info("Building FAISS Vector Database...")
        # Note: In a large system, you'd retry this too, but for 5 chunks it's usually safe
        response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=KNOWLEDGE_CHUNKS
        )
        
        embeddings = np.array([emb.values for emb in response.embeddings], dtype=np.float32)
        
        # gemini-embedding-001 produces 768 or 3072 dims depending on config; default is 768
        dimension_size = len(embeddings[0]) 
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
    if not genai_client or not vector_db:
        raise HTTPException(status_code=500, detail="Database or Client not initialized")
    
    try:
        # 1. Embed the user's question
        question_embedding = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=request.query
        )
        question_vector = np.array([question_embedding.embeddings[0].values], dtype=np.float32)
        
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
        
        initial_res = call_gemini_with_retry('gemini-2.5-flash-lite', initial_prompt)
        initial_answer = initial_res.text

        # 4. Step 2: Agentic Critique & Hallucination Check
        critique_prompt = f"""
        As an AI Auditor, evaluate if this answer is strictly grounded in the provided context.
        Context: {retrieved_context}
        Answer: {initial_answer}

        If the answer contains hallucinations or info not in the context, rewrite it to be 100% faithful.
        Otherwise, return the original answer exactly.
        """

        final_res = call_gemini_with_retry('gemini-2.5-flash-lite', critique_prompt)

        return QueryResponse(answer=final_res.text, context=retrieved_context)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))