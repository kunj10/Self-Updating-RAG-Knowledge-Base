import os
import pytest
import json
import time
from fastapi.testclient import TestClient
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import GeminiModel
from app.main import app

def load_golden_data():
    with open("golden_dataset.json", "r") as f:
        return json.load(f)

@pytest.fixture
def client():
    # Use the context manager to trigger FastAPI lifespan (DB init)
    with TestClient(app) as c:
        time.sleep(2)
        yield c

@pytest.mark.parametrize("case", load_golden_data())
def test_rag_api_relevancy_and_faithfulness(client, case):
    time.sleep(5)
    
    question = case["input"]
    
    # Hit our FastAPI endpoint
    response = client.post("/ask", json={"query": question})
    assert response.status_code == 200, f"API Error: {response.text}"
    
    data = response.json()
    actual_answer = data["answer"]
    retrieval_context = [data["context"]]
    
    # 3. Setup Gemini Judge
    eval_model = GeminiModel(model="gemini-2.5-flash-lite")
    
    # 4. Define DeepEval Metrics
    # CRITICAL: Set async_mode=False to prevent DeepEval from firing 
    # multiple evaluation requests at the same time.
    faithfulness = FaithfulnessMetric(
        threshold=0.7, 
        model=eval_model,
        async_mode=False 
    )
    
    # 5. Create the LLM Test Case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        retrieval_context=retrieval_context
    )
    
    # 6. Run the Evaluation
    assert_test(test_case, [faithfulness])