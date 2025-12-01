import re
from typing import Dict, Any
from human_eval.execution import check_correctness

def clean_code(completion: str) -> str:
    """
    Extracts python code from a completion string that might contain markdown.
    """
    # 1. Try to find markdown code blocks with python tag
    code_block_pattern = r"```python\n(.*?)\n```"
    match = re.search(code_block_pattern, completion, re.DOTALL)
    if match:
        return match.group(1)
    
    # 2. Try to find generic markdown code blocks
    code_block_pattern_2 = r"```\n(.*?)\n```"
    match = re.search(code_block_pattern_2, completion, re.DOTALL)
    if match:
        return match.group(1)

    # 3. If no code blocks, return as is (GPT-3.5 usually returns pure code)
    return completion

def evaluate_functional_correctness(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the functional correctness of a sample.
    Sample must contain:
    - task_id
    - completion
    - test (from the problem definition)
    - entry_point
    
    Returns a dictionary with 'passed' (bool) and 'result' (str).
    """
    problem = {
        "task_id": sample["task_id"],
        "prompt": sample["prompt"],
        "test": sample["test"],
        "entry_point": sample["entry_point"]
    }
    
    try:
        # Clean the completion before execution
        cleaned_completion = clean_code(sample["completion"])
        result = check_correctness(problem, cleaned_completion, timeout=3.0)
        return {
            "passed": result["passed"],
            "result": result["result"]
        }
    except Exception as e:
        return {
            "passed": False,
            "result": f"Error during execution: {str(e)}"
        }
