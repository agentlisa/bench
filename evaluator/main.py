import dotenv
import os
from evaluator import OpenAIEvaluator
from judge import Judge
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import json
import tqdm

dotenv.load_dotenv()

def data_loader_top_500():
    with open("subsets/random_500_high_risk_ids.json", "r") as f:
        import json
        ids = json.load(f)
    for report_id in ids:
        with open(f"dataset/{report_id}.json", "r", encoding="utf-8") as f:
            item = json.load(f)

            code = item["Code"]
            code_str = "\n\n".join([x["filename"] + ":\n" + x["content"] for x in code])

            ground_truth = item["Content"]

            yield code_str, ground_truth


def check_one_case(code: str, gt:str) -> tuple[bool, dict]:
    evaluator = OpenAIEvaluator(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_name=os.getenv("MODEL_NAME"),
    )
    submission, eval_tokens = evaluator.evaluate(code)
    judge = Judge()
    result, judge_tokens = judge.judge(submission, gt)
    
    # Combine token usage
    total_tokens = {
        'evaluator': eval_tokens,
        'judge': judge_tokens,
        'total_prompt_tokens': eval_tokens.get('prompt_tokens', 0) + judge_tokens.get('prompt_tokens', 0),
        'total_completion_tokens': eval_tokens.get('completion_tokens', 0) + judge_tokens.get('completion_tokens', 0),
        'total_tokens': eval_tokens.get('total_tokens', 0) + judge_tokens.get('total_tokens', 0)
    }
    
    return result, total_tokens



def run_evaluation():
    total_cases = 0
    correct_cases = 0
    
    # Token usage tracking
    total_eval_prompt_tokens = 0
    total_eval_completion_tokens = 0
    total_eval_tokens = 0
    total_judge_prompt_tokens = 0
    total_judge_completion_tokens = 0
    total_judge_tokens = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_case: dict[Future, tuple[str, str]] = {}
        for code, gt in data_loader_top_500():
            future = executor.submit(check_one_case, code, gt)
            future_to_case[future] = (code, gt)

        for future in tqdm.tqdm(as_completed(future_to_case), total=len(future_to_case), desc="Evaluating Cases"):
            code, gt = future_to_case[future]
            try:
                result, token_usage = future.result()
                total_cases += 1
                if result:
                    correct_cases += 1
                
                # Accumulate token usage
                eval_tokens = token_usage.get('evaluator', {})
                judge_tokens = token_usage.get('judge', {})
                
                total_eval_prompt_tokens += eval_tokens.get('prompt_tokens', 0)
                total_eval_completion_tokens += eval_tokens.get('completion_tokens', 0)
                total_eval_tokens += eval_tokens.get('total_tokens', 0)
                
                total_judge_prompt_tokens += judge_tokens.get('prompt_tokens', 0)
                total_judge_completion_tokens += judge_tokens.get('completion_tokens', 0)
                total_judge_tokens += judge_tokens.get('total_tokens', 0)
                
                print(f"Current Accuracy: {correct_cases}/{total_cases} = {correct_cases/total_cases:.2%} | "
                      f"Tokens: {total_eval_tokens + total_judge_tokens:,}")
            except Exception as e:
                print(f"Error processing case: {e}")

    # Print final statistics
    print("\n" + "="*80)
    print(f"Final Accuracy: {correct_cases}/{total_cases} = {correct_cases/total_cases:.2%}")
    print("\nToken Usage Summary:")
    print("-"*80)
    print(f"Evaluator (Model: {os.getenv('MODEL_NAME')}):")
    print(f"  Prompt Tokens:     {total_eval_prompt_tokens:,}")
    print(f"  Completion Tokens: {total_eval_completion_tokens:,}")
    print(f"  Total Tokens:      {total_eval_tokens:,}")
    print(f"\nJudge (gpt-5.1-nano):")
    print(f"  Prompt Tokens:     {total_judge_prompt_tokens:,}")
    print(f"  Completion Tokens: {total_judge_completion_tokens:,}")
    print(f"  Total Tokens:      {total_judge_tokens:,}")
    print(f"\nGrand Total:")
    print(f"  Prompt Tokens:     {total_eval_prompt_tokens + total_judge_prompt_tokens:,}")
    print(f"  Completion Tokens: {total_eval_completion_tokens + total_judge_completion_tokens:,}")
    print(f"  Total Tokens:      {total_eval_tokens + total_judge_tokens:,}")
    print("="*80)



def main():
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set in environment variables."
    assert os.getenv("OPENAI_BASE_URL") is not None, "OPENAI_BASE_URL is not set in environment variables."
    assert os.getenv("MODEL_NAME") is not None, "MODEL_NAME is not set in environment variables."

    run_evaluation()

if __name__ == "__main__":
    main()
