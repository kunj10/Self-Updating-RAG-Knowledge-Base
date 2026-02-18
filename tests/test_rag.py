import os
import pytest
import json
import time
from fastapi.testclient import TestClient
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from groq import Groq
from app.main import app


# --- Custom Groq Judge Wrapper for DeepEval ---

class GroqJudgeModel(DeepEvalBaseLLM):
    """Wraps Groq's API so DeepEval can use Llama-3.3-70B as the evaluation judge."""

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval requires this; we just call the sync version
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# --- Test Helpers ---

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

    # 3. Setup Groq Judge (Llama-3.3-70B — much more capable than small models)
    eval_model = GroqJudgeModel()

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