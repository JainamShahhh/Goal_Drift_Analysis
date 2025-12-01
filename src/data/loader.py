from typing import Iterable, Dict, Any
from human_eval.data import read_problems

def load_humaneval() -> Iterable[Dict[str, Any]]:
    """
    Loads the HumanEval dataset.
    Returns an iterable of problem dictionaries.
    Each dictionary contains:
        - task_id: str
        - prompt: str
        - entry_point: str
        - canonical_solution: str
        - test: str
    """
    problems = read_problems()
    for task_id, problem in problems.items():
        yield problem

if __name__ == "__main__":
    # Test the loader
    print("Loading HumanEval...")
    count = 0
    for problem in load_humaneval():
        count += 1
    print(f"Loaded {count} problems.")
