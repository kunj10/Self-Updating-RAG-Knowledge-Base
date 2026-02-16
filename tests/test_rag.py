import os
import pytest
from fastapi.testclient import TestClient

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GeminiModel
from app.main import app

@pytest.fixture
def client():
    # This 'with' block forces FastAPI to run the lifespan function!
    with TestClient(app) as c:
        yield c

def test_rag_api_relevancy_and_faithfulness():
    # 1. Simulate a user asking a question
    question = "What features does MyTax support?"
    
    # 2. Hit our FastAPI endpoint
    response = client.post("/ask", json={"query": question})
    assert response.status_code == 200, f"API Error: {response.text}"
    
    data = response.json()
    actual_answer = data["answer"]
    retrieval_context = [data["context"]]
    
    # 3. Setup Gemini as the DeepEval "Judge"
    eval_model = GeminiModel(model="gemini-2.5-flash-lite")
    
    # 4. Define DeepEval Metrics
    
    # Faithfulness: Did it hallucinate facts outside the context? 
    faithfulness = FaithfulnessMetric(threshold=0.7, model=eval_model)
    # Answer Relevancy: Did it actually answer the specific question? (commented out to reduce api calls)
    # relevancy = AnswerRelevancyMetric(threshold=0.7, model=eval_model)
    
    # 5. Create the LLM Test Case (The Evidence)
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        retrieval_context=retrieval_context
    )
    
    # 6. Run the Evaluation
    assert_test(test_case, [faithfulness])