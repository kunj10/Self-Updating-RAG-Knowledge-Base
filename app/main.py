import os
import numpy as np
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

vector_db = None
genai_client = None

KNOWLEDGE_CHUNKS = [
    "MyTax is a modern GST compliance and billing software designed for Indian businesses.",
    "MyTax supports automated GST return filing.",
    "MyTax supports e-invoicing.",
    "MyTax allows multi-user access for enterprise teams.",
    "The MyTax support team can be reached at support@mytax.com."
]

# 2. This runs automatically when the FastAPI server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, genai_client
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai_client = genai.Client(api_key=api_key)
        
        print("Building FAISS Vector Database...")
        response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=KNOWLEDGE_CHUNKS
        )
        
        embeddings = np.array([emb.values for emb in response.embeddings], dtype=np.float32)
        
        dimension_size = 3072
        vector_db = faiss.IndexFlatL2(dimension_size)
        vector_db.add(embeddings)
        print(f"Successfully loaded {vector_db.ntotal} documents into the Vector DB.")
    
    yield

app = FastAPI(title="MyTax AutoRAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not genai_client or not vector_db:
        raise HTTPException(status_code=500, detail="Database or Client not initialized")
        
    # 3. Embed the user's question into numbers
    question_embedding = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=request.query
    )
    question_vector = np.array([question_embedding.embeddings[0].values], dtype=np.float32)
    
    # 4. Search FAISS for the top 2 most relevant chunks
    distances, indices = vector_db.search(question_vector, k=2)
    
    # 5. Extract the matching text blocks
    retrieved_chunks = [KNOWLEDGE_CHUNKS[i] for i in indices[0]]
    retrieved_context = "\n".join(retrieved_chunks)
    
    # 6. Send ONLY the relevant chunks to the LLM
    prompt = f"""
    You are a helpful customer support AI for MyTax.
    Answer the user's question based ONLY on the provided context. 
    If you don't know the answer based on the context, say "I don't know."
    
    Context:
    {retrieved_context}
    
    Question:
    {request.query}
    """
    
    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash-lite', 
            contents=prompt,
        )
        return QueryResponse(answer=response.text, context=retrieved_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))