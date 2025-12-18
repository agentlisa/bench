from langchain_openai import ChatOpenAI
from langchain_core.callbacks import UsageMetadataCallbackHandler 
from pydantic import BaseModel, Field
import os


class JudgeResult(BaseModel):
    result: bool = Field(..., description="Whether the submission contains the ground truth.")

class Judge:
    def __init__(self, model_name: str = "gpt-5-nano") -> None:
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model_name = model_name
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def judge(self, submission: str, ground_truth: str) -> tuple[bool, dict]:
        prompt = f"""
        You are a judge evaluating the quality of a submission against the ground truth. The submission is an audit report from a smart contract security analysis tool, and the ground truth is a human-written audit report.
        You just need to determine whether the submission contains the ground truth.

        Submission:
        {submission}

        Ground Truth:
        {ground_truth}

        """
        callback = UsageMetadataCallbackHandler() 
        response = self.llm.with_structured_output(JudgeResult).invoke(prompt, config={"callbacks": [callback]})
        
        # Extract token usage from response
        token_usage = {
            'input_tokens': callback.usage_metadata.get(self.model_name).get('input_tokens', 0),
            'output_tokens': callback.usage_metadata.get(self.model_name).get('output_tokens', 0),
            'total_tokens': callback.usage_metadata.get(self.model_name).get('total_tokens', 0)
        }
        self.input_tokens += token_usage['input_tokens']
        self.output_tokens += token_usage['output_tokens']
        self.total_tokens += token_usage['total_tokens']
        
        return response.result, token_usage


