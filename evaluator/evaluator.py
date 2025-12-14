from langchain_openai import ChatOpenAI
import os


PREDICT_PROMPT = """
Given the following smart contract, evaluate its security and identify any potential vulnerabilities.
"""


class Evaluator:
    def __init__(self, api_key:str) -> None:
        self.api_key = api_key


class OpenAIEvaluator(Evaluator):
    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        super().__init__(api_key)
        self.base_url = base_url
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model = self.model_name,
            api_key = self.api_key,
            base_url = self.base_url,
        )
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def evaluate(self, code: str) -> tuple[str, dict]:
        response = self.llm.invoke(PREDICT_PROMPT + "\n\n" + code)
        
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
        
        return response.content, token_usage

