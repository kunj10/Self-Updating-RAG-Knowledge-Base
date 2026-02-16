import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="AutoRAG API")

# Hardcoded knowledge base for V1 (We will move this to S3/VectorDB later)
KNOWLEDGE_BASE = """
MyTax is a modern GST compliance and billing software designed for Indian businesses.
It supports automated GST return filing, e-invoicing, and multi-user access.
The support team can be reached at support@vypartaxone.com.
"""

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    # The API key is securely injected by GitHub Actions during testing
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_KEY")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are a helpful customer support AI.
    Answer the user's question based ONLY on the provided context. 
    If you don't know the answer based on the context, say "I don't know."
    
    Context:
    {KNOWLEDGE_BASE}
    
    Question:
    {request.query}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,    
        )
        return QueryResponse(answer=response.text, context=KNOWLEDGE_BASE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))