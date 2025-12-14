from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os


class JudgeResult(BaseModel):
    result: bool = Field(..., description="Whether the submission contains the ground truth.")

class Judge:
    def __init__(self) -> None:
        model_name = "gpt-5-nano"
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def judge(self, submission: str, ground_truth: str) -> tuple[bool, dict]:
        prompt = f"""
        You are a judge evaluating the quality of a submission against the ground truth. The submission is an audit report from a smart contract security analysis tool, and the ground truth is a human-written audit report.
        You just need to determine whether the submission contains the ground truth.

        Submission:
        {submission}

        Ground Truth:
        {ground_truth}

        """
        response = self.llm.with_structured_output(JudgeResult).invoke(prompt)
        
        # Extract token usage from response
        token_usage = {}
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            token_usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            self.prompt_tokens += token_usage['prompt_tokens']
            self.completion_tokens += token_usage['completion_tokens']
            self.total_tokens += token_usage['total_tokens']
        
        return response.result, token_usage


