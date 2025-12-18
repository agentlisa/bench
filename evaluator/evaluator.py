from langchain_openai import ChatOpenAI
from langchain_core.callbacks import UsageMetadataCallbackHandler 
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
        self.input_tokens = 0
        self.output_tokens = 0

    def evaluate(self, code: str) -> tuple[str, dict]:
        
        callback = UsageMetadataCallbackHandler() 
        response = self.llm.invoke(PREDICT_PROMPT + "\n\n" + code, config={"callbacks": [callback]})
        
        
        # Extract token usage from response
        token_usage = {
            'input_tokens': callback.usage_metadata.get(self.model_name).get('input_tokens', 0),
            'output_tokens': callback.usage_metadata.get(self.model_name).get('output_tokens', 0),
            'total_tokens': callback.usage_metadata.get(self.model_name).get('total_tokens', 0)
        }
        self.input_tokens += token_usage['input_tokens']
        self.output_tokens += token_usage['output_tokens']
        self.total_tokens += token_usage['total_tokens']
        
        return response.content, token_usage

