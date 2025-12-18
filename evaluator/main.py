import dotenv
import os
from evaluator import OpenAIEvaluator
from judge import Judge
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError as FutureTimeoutError
import json
import tqdm
import time
import datetime

dotenv.load_dotenv()

# 超时配置
CASE_TIMEOUT_SECONDS = 5 * 60  # 5分钟

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


def _execute_check_one_case(code: str, gt:str, model_name:str) -> tuple[bool, dict]:
    """内部执行函数，不带超时控制"""
    start_time = time.time()
    
    evaluator = OpenAIEvaluator(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_name=model_name,
    )
    submission, eval_tokens = evaluator.evaluate(code)
    judge = Judge(os.getenv("JUDGE_MODEL_NAME", "openai/gpt-5-nano"))
    result, judge_tokens = judge.judge(submission, gt)
    
    elapsed_time = time.time() - start_time
    
    # Combine token usage
    total_tokens = {
        'evaluator': eval_tokens,
        'judge': judge_tokens,
        'total_input_tokens': eval_tokens.get('input_tokens', 0) + judge_tokens.get('input_tokens', 0),
        'total_output_tokens': eval_tokens.get('output_tokens', 0) + judge_tokens.get('output_tokens', 0),
        'total_tokens': eval_tokens.get('total_tokens', 0) + judge_tokens.get('total_tokens', 0),
        'elapsed_time': elapsed_time
    }
    
    return result, total_tokens


def check_one_case_with_timeout(code: str, gt: str, model_name: str) -> tuple[bool, dict]:
    """带超时控制的check_one_case，使用独立线程池执行并强制超时"""
    with ThreadPoolExecutor(max_workers=1) as timeout_executor:
        future = timeout_executor.submit(_execute_check_one_case, code, gt, model_name)
        try:
            result, token_usage = future.result(timeout=CASE_TIMEOUT_SECONDS)
            return result, token_usage
        except FutureTimeoutError:
            # 超时后尝试取消future
            future.cancel()
            raise TimeoutError(f"Case execution exceeded {CASE_TIMEOUT_SECONDS}s timeout")



def run_evaluation(model_name: str):
    total_cases = 0
    correct_cases = 0
    
    # Token usage tracking
    total_eval_input_tokens = 0
    total_eval_output_tokens = 0
    total_eval_tokens = 0
    total_judge_input_tokens = 0
    total_judge_output_tokens = 0
    total_judge_tokens = 0

    if os.path.exists(f"evaluation_result_{model_name.replace('/', '_')}_summary.json"):
        print(f"Evaluation for model {model_name} already exists. Skipping...")
        return

    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_case: dict[Future, tuple[str, str]] = {}
        for code, gt in data_loader_top_500():
            future = executor.submit(check_one_case_with_timeout, code, gt, model_name)
            future_to_case[future] = (code, gt)

        for future in tqdm.tqdm(as_completed(future_to_case), total=len(future_to_case), desc=f"Evaluating Cases ({model_name})"):
            code, gt = future_to_case[future]
            try:
                result, token_usage = future.result()
                total_cases += 1
                if result:
                    correct_cases += 1
                
                # Accumulate token usage
                eval_tokens = token_usage.get('evaluator', {})
                judge_tokens = token_usage.get('judge', {})
                
                total_eval_input_tokens += eval_tokens.get('input_tokens', 0)
                total_eval_output_tokens += eval_tokens.get('output_tokens', 0)
                total_eval_tokens += eval_tokens.get('total_tokens', 0)
                
                total_judge_input_tokens += judge_tokens.get('input_tokens', 0)
                total_judge_output_tokens += judge_tokens.get('output_tokens', 0)
                total_judge_tokens += judge_tokens.get('total_tokens', 0)
                
                elapsed = token_usage.get('elapsed_time', 0)
                print(f"Current Accuracy: {correct_cases}/{total_cases} = {correct_cases/total_cases:.2%} | "
                      f"Tokens: {total_eval_tokens + total_judge_tokens:,} | Time: {elapsed:.1f}s | Now: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except TimeoutError as e:
                print(f"⏱️  Case timeout after {CASE_TIMEOUT_SECONDS}s - skipping")
                total_cases += 1  # Count the case as failed
            except Exception as e:
                print(f"❌ Error processing case: {e}")
                total_cases += 1  # Still count the case even if it failed
            # dump intermediate result for each step
            intermediate_result = {
                "current_accuracy": {
                    "correct_cases": correct_cases,
                    "total_cases": total_cases,
                    "accuracy_percentage": correct_cases / total_cases if total_cases > 0 else 0
                },
                "token_usage": {
                    "evaluator": {
                        "input_tokens": total_eval_input_tokens,
                        "output_tokens": total_eval_output_tokens,
                        "total_tokens": total_eval_tokens
                    },
                    "judge": {
                        "input_tokens": total_judge_input_tokens,
                        "output_tokens": total_judge_output_tokens,
                        "total_tokens": total_judge_tokens
                    },
                    "grand_total": {
                        "input_tokens": total_eval_input_tokens + total_judge_input_tokens,
                        "output_tokens": total_eval_output_tokens + total_judge_output_tokens,
                        "total_tokens": total_eval_tokens + total_judge_tokens
                    }
                }
            }
            with open(f"evaluation_result_{model_name.replace('/', '_')}_intermediate.json", "w") as f:
                json.dump(intermediate_result, f, indent=4)


    # Print final statistics
    print("\n" + "="*80)
    print(f"Final Accuracy: {correct_cases}/{total_cases} = {correct_cases/total_cases:.2%}")
    print("\nToken Usage Summary:")
    print("-"*80)
    print(f"Evaluator (Model: {os.getenv('MODEL_NAME')}):")
    print(f"  Prompt Tokens:     {total_eval_input_tokens:,}")
    print(f"  Completion Tokens: {total_eval_output_tokens:,}")
    print(f"  Total Tokens:      {total_eval_tokens:,}")
    print(f"\nJudge (gpt-5.1-nano):")
    print(f"  Prompt Tokens:     {total_judge_input_tokens:,}")
    print(f"  Completion Tokens: {total_judge_output_tokens:,}")
    print(f"  Total Tokens:      {total_judge_tokens:,}")
    print(f"\nGrand Total:")
    print(f"  Prompt Tokens:     {total_eval_input_tokens + total_judge_input_tokens:,}")
    print(f"  Completion Tokens: {total_eval_output_tokens + total_judge_output_tokens:,}")
    print(f"  Total Tokens:      {total_eval_tokens + total_judge_tokens:,}")
    print("="*80)

    os.remove(f"evaluation_result_{model_name.replace('/', '_')}_intermediate.json")

    # Save the result to a file, including token usage summary
    result_summary = {
        "final_accuracy": {
            "correct_cases": correct_cases,
            "total_cases": total_cases,
            "accuracy_percentage": correct_cases / total_cases if total_cases > 0 else 0
        },
        "token_usage": {
            "evaluator": {
                "input_tokens": total_eval_input_tokens,
                "output_tokens": total_eval_output_tokens,
                "total_tokens": total_eval_tokens
            },
            "judge": {
                "input_tokens": total_judge_input_tokens,
                "output_tokens": total_judge_output_tokens,
                "total_tokens": total_judge_tokens
            },
            "grand_total": {
                "input_tokens": total_eval_input_tokens + total_judge_input_tokens,
                "output_tokens": total_eval_output_tokens + total_judge_output_tokens,
                "total_tokens": total_eval_tokens + total_judge_tokens
            }
        }
    }

    with open(f"evaluation_result_{model_name.replace('/', '_')}_summary.json", "w") as f:
        json.dump(result_summary, f, indent=4)



def main():
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set in environment variables."
    assert os.getenv("OPENAI_BASE_URL") is not None, "OPENAI_BASE_URL is not set in environment variables."
    assert os.getenv("MODEL_NAMES") is not None, "MODEL_NAME is not set in environment variables."

    for model_name in os.getenv("MODEL_NAMES").split(";"):
        run_evaluation(model_name)

if __name__ == "__main__":
    main()
