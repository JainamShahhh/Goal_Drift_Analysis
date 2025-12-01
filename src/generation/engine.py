import json
import os
import argparse
from tqdm import tqdm
from src.data.loader import load_humaneval
from src.data.prompts import PromptCondition, apply_condition
from src.generation.models import get_model

def run_generation(provider: str, model_name: str, output_file: str, iterations: int = 10, limit: int = None):
    """
    Runs the generation pipeline.
    """
    print(f"Initializing {provider} model: {model_name}")
    model = get_model(provider, model_name)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Loading HumanEval...")
    problems = list(load_humaneval())
    if limit:
        problems = problems[:limit]
        print(f"Limiting to first {limit} tasks.")
    
    with open(output_file, "a") as f:
        for problem in tqdm(problems, desc="Generating"):
            task_id = problem["task_id"]
            original_prompt = problem["prompt"]
            
            for condition in PromptCondition:
                # Apply prompt condition
                final_prompt = apply_condition(original_prompt, condition)
                
                # Generate 'iterations' samples
                # We can do this in one batch for OpenAI
                completions = model.generate(final_prompt, n=iterations, temperature=0.7)
                
                for i, code in enumerate(completions):
                    result = {
                        "task_id": task_id,
                        "condition": condition.value,
                        "iteration": i,
                        "prompt": final_prompt,
                        "completion": code,
                        "model": model_name,
                        "canonical_solution": problem["canonical_solution"],
                        "entry_point": problem["entry_point"]
                    }
                    f.write(json.dumps(result) + "\n")
                    f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, required=True, choices=["openai", "anthropic", "google"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/results.jsonl")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tasks for testing")
    
    args = parser.parse_args()
    run_generation(args.provider, args.model, args.output, args.iterations, args.limit)
