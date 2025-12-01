import os
import json
from src.generation.engine import run_generation
from src.analysis.analyze import analyze_results

def merge_results(output_file, input_files):
    print(f"Merging {input_files} into {output_file}...")
    seen = set()
    with open(output_file, "w") as outfile:
        for infile in input_files:
            if not os.path.exists(infile):
                print(f"Warning: {infile} does not exist. Skipping.")
                continue
            
            with open(infile, "r") as f:
                for line in f:
                    # Deduplicate based on task_id + condition + model + iteration
                    try:
                        data = json.loads(line)
                        key = (data["task_id"], data["condition"], data["model"], data["iteration"])
                        if key not in seen:
                            outfile.write(line)
                            seen.add(key)
                    except:
                        continue
    print(f"Merged {len(seen)} unique records.")

def main():
    print("=== Starting Overnight Experiment ===")
    
    # 1. Run Anthropic (Claude 3.5 Sonnet)
    # print("\n--- Step 1: Generating with Anthropic (Claude 3.5) ---")
    # try:
    #     run_generation(
    #         provider="anthropic", 
    #         model_name="claude-3-haiku-20240307", 
    #         output_file="data/results_anthropic.jsonl", 
    #         iterations=10
    #     )
    # except Exception as e:
    #     print(f"Anthropic generation failed: {e}")

    # 2. Merge Results (OpenAI + Anthropic)
    print("\n--- Step 2: Merging Results ---")
    files_to_merge = [
        "data/results_full.jsonl",       # Existing GPT-3.5 data
        "data/results_anthropic.jsonl",  # New Anthropic data
    ]
    merge_results("data/results_combined.jsonl", files_to_merge)

    # 4. Run Analysis
    print("\n--- Step 4: Running Final Analysis ---")
    try:
        analyze_results(
            input_file="data/results_combined.jsonl", 
            output_dir="analysis_combined"
        )
    except Exception as e:
        print(f"Analysis failed: {e}")

    print("\n=== Overnight Experiment Complete! ===")

if __name__ == "__main__":
    main()
