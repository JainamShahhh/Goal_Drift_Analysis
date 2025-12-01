import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from src.evaluation.executor import evaluate_functional_correctness
from src.evaluation.metrics import calculate_pass_at_k, calculate_goal_drift

def analyze_results(input_file: str, output_dir: str):
    """
    Analyzes the results JSONL file.
    """
    print(f"Loading results from {input_file}...")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # 1. Evaluate Correctness
    # Check if we already have evaluated results to save time
    eval_cache_file = f"{output_dir}/evaluated_results.csv"
    if os.path.exists(eval_cache_file):
        print(f"Loading cached evaluation results from {eval_cache_file}...")
        df = pd.read_csv(eval_cache_file)
    else:
        print("Evaluating functional correctness...")
        # Load HumanEval once
        from src.data.loader import load_humaneval
        problems = {p["task_id"]: p for p in load_humaneval()}
        
        # Prepare samples for parallel execution
        samples = []
        for _, row in df.iterrows():
            problem = problems[row["task_id"]]
            sample = {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "completion": row["completion"],
                "test": problem["test"],
                "entry_point": problem["entry_point"]
            }
            samples.append(sample)
    
        print(f"Evaluating {len(samples)} samples in parallel...")
        
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        
        eval_results = []
        with ProcessPoolExecutor() as executor:
            results_iter = list(tqdm(executor.map(evaluate_functional_correctness, samples), total=len(samples)))
            
        for res in results_iter:
            eval_results.append(res["passed"])
        
        df["passed"] = eval_results
        
        # Calculate Code Length (Verbosity)
        df["code_length"] = df["completion"].apply(len)
        
        # Save to cache
        print(f"Saving evaluated results to {eval_cache_file}...")
        df.to_csv(eval_cache_file, index=False)
    
    # 2. Calculate Pass@1 per Condition
    print("Calculating Pass@1...")
    pass_rates = df.groupby(["model", "condition"])["passed"].mean().reset_index()
    pass_rates.rename(columns={"passed": "pass_rate"}, inplace=True)
    print("Overall Pass Rates:")
    print(pass_rates)
    pass_rates.to_csv(f"{output_dir}/pass_rates_summary.csv", index=False)

    # B. Per-Task Analysis (Detailed)
    task_stats = df.groupby(["model", "condition", "task_id"])["passed"].mean().reset_index()
    task_stats.rename(columns={"passed": "task_pass_rate"}, inplace=True)
    task_stats.to_csv(f"{output_dir}/per_task_breakdown.csv", index=False)

    # 3. Statistical Significance (Mann-Whitney U)
    print("\nCalculating Statistical Significance (Mann-Whitney U)...")
    from scipy.stats import mannwhitneyu
    
    stats_results = []
    neutral_scores = task_stats[task_stats["condition"] == "neutral"]["task_pass_rate"]
    
    for cond in ["speed", "caution", "reputation"]:
        cond_scores = task_stats[task_stats["condition"] == cond]["task_pass_rate"]
        if len(cond_scores) > 0:
            u_stat, p_val = mannwhitneyu(neutral_scores, cond_scores, alternative='two-sided')
            stats_results.append({
                "comparison": f"neutral vs {cond}",
                "u_stat": u_stat,
                "p_value": p_val,
                "significant": p_val < 0.05
            })
    
    stats_df = pd.DataFrame(stats_results)
    print(stats_df)
    stats_df.to_csv(f"{output_dir}/statistical_significance.csv", index=False)

    # C. Reliability Analysis
    perfect_tasks = task_stats[task_stats["task_pass_rate"] == 1.0].groupby(["model", "condition"]).size().reset_index(name="perfect_count")
    total_tasks = len(df["task_id"].unique())
    perfect_tasks["reliability_score"] = perfect_tasks["perfect_count"] / total_tasks
    perfect_tasks.to_csv(f"{output_dir}/reliability_scores.csv", index=False)

    # D. Drift Analysis Table
    drift_data = []
    for model in pass_rates["model"].unique():
        neutral_rate = pass_rates[(pass_rates["model"] == model) & (pass_rates["condition"] == "neutral")]["pass_rate"].values[0]
        for cond in ["speed", "caution", "reputation"]:
            cond_rate = pass_rates[(pass_rates["model"] == model) & (pass_rates["condition"] == cond)]["pass_rate"].values[0]
            drift = neutral_rate - cond_rate
            drift_pct = (drift / neutral_rate) * 100 if neutral_rate > 0 else 0
            drift_data.append({
                "model": model,
                "condition": cond,
                "neutral_rate": neutral_rate,
                "condition_rate": cond_rate,
                "absolute_drift": drift,
                "relative_drift_pct": drift_pct
            })
    drift_df = pd.DataFrame(drift_data)
    drift_df.to_csv(f"{output_dir}/drift_analysis.csv", index=False)

    # E. Top 10 Drifted Tasks
    # Calculate drift per task for Speed condition
    neutral_tasks = task_stats[task_stats["condition"] == "neutral"][["task_id", "model", "task_pass_rate"]]
    speed_tasks = task_stats[task_stats["condition"] == "speed"][["task_id", "model", "task_pass_rate"]]
    merged_tasks = pd.merge(neutral_tasks, speed_tasks, on=["task_id", "model"], suffixes=("_neutral", "_speed"))
    merged_tasks["drift"] = merged_tasks["task_pass_rate_neutral"] - merged_tasks["task_pass_rate_speed"]
    top_drift = merged_tasks.sort_values("drift", ascending=False).head(20)
    top_drift.to_csv(f"{output_dir}/top_drifted_tasks.csv", index=False)

    # 4. Visualizations
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Bar Plot (Mean Pass Rate)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pass_rates, x="condition", y="pass_rate", hue="model", palette="viridis")
    plt.title("Average Pass@1 Rate by Condition")
    plt.ylabel("Pass Rate")
    plt.ylim(0, 1.0)
    plt.savefig(f"{output_dir}/plot_1_bar_pass_rate.png")
    plt.close()

    # Plot 2: Box Plot (Robustness)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=task_stats, x="condition", y="task_pass_rate", hue="model", palette="Set2")
    plt.title("Distribution of Pass Rates Across Tasks (Robustness)")
    plt.ylabel("Pass Rate per Task (0.0 - 1.0)")
    plt.savefig(f"{output_dir}/plot_2_box_robustness.png")
    plt.close()

    # Plot 3: Violin Plot (Density)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=task_stats, x="condition", y="task_pass_rate", hue="model", split=True, inner="quart")
    plt.title("Density of Pass Rates (Violin Plot)")
    plt.savefig(f"{output_dir}/plot_3_violin_density.png")
    plt.close()

    # Plot 4: Reliability Plot (Bar)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=perfect_tasks, x="condition", y="reliability_score", hue="model", palette="magma")
    plt.title("Reliability Score (Fraction of Perfectly Solved Tasks)")
    plt.ylabel("Fraction of Tasks with 10/10 Passes")
    plt.ylim(0, 1.0)
    plt.savefig(f"{output_dir}/plot_4_bar_reliability.png")
    plt.close()

    # Plot 6: Scatter Plot (Verbosity vs Correctness)
    # We need code length. If loaded from cache, it's there. If not, we computed it.
    if "code_length" not in df.columns:
         df["code_length"] = df["completion"].apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="code_length", y="passed", hue="condition", alpha=0.1)
    # Passed is boolean 0/1, so scatter might look weird. 
    # Better: Bin by length and show pass rate.
    # Or: Scatter of Task Pass Rate vs Avg Length
    task_len = df.groupby(["model", "condition", "task_id"])["code_length"].mean().reset_index()
    task_combined = pd.merge(task_stats, task_len, on=["model", "condition", "task_id"])
    
    sns.scatterplot(data=task_combined, x="code_length", y="task_pass_rate", hue="condition", style="model", alpha=0.6)
    plt.title("Code Verbosity vs. Correctness")
    plt.xlabel("Average Code Length (chars)")
    plt.ylabel("Pass Rate")
    plt.savefig(f"{output_dir}/plot_6_scatter_verbosity.png")
    plt.close()

    # Plot 7: Heatmap of Task Pass Rates (First 50 tasks)
    # Pivot for heatmap: Rows=Task, Cols=Condition
    # Filter for one model (e.g., Claude) to keep it readable
    heatmap_data = task_stats[task_stats["model"].str.contains("claude")].pivot(index="task_id", columns="condition", values="task_pass_rate")
    # Sort by neutral pass rate
    heatmap_data = heatmap_data.sort_values("neutral", ascending=False).head(40) # Top 40 tasks
    
    plt.figure(figsize=(8, 12))
    sns.heatmap(heatmap_data, cmap="RdYlGn", annot=False)
    plt.title("Pass Rate Heatmap (Claude 3 Haiku - Top 40 Tasks)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_7_heatmap_tasks.png")
    plt.close()

    # E. CodeBLEU Analysis (Calculated if not cached)
    if "codebleu" not in df.columns:
        print("\nCalculating CodeBLEU Proxy (Semantic Similarity)...")
        from src.evaluation.metrics import calculate_codebleu_proxy
        # Load problems if not loaded
        if 'problems' not in locals():
             from src.data.loader import load_humaneval
             problems = {p["task_id"]: p for p in load_humaneval()}
             
        bleu_scores = []
        for _, row in df.iterrows():
            problem = problems[row["task_id"]]
            score = calculate_codebleu_proxy(problem["canonical_solution"], row["completion"])
            bleu_scores.append(score)
        df["codebleu"] = bleu_scores
        # Update cache
        df.to_csv(eval_cache_file, index=False)
    
    # Aggregate CodeBLEU
    bleu_stats = df.groupby(["model", "condition"])["codebleu"].mean().reset_index()
    bleu_stats.to_csv(f"{output_dir}/codebleu_scores.csv", index=False)
    
    # Plot 5: CodeBLEU Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bleu_stats, x="condition", y="codebleu", hue="model", palette="coolwarm")
    plt.title("Average CodeBLEU Score (Semantic Similarity)")
    plt.ylabel("CodeBLEU Score (0.0 - 1.0)")
    plt.ylim(0, 1.0)
    plt.savefig(f"{output_dir}/plot_5_bar_codebleu.png")
    plt.close()

    print(f"Detailed analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/results.jsonl")
    parser.add_argument("--output", type=str, default="analysis_results")
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output, exist_ok=True)
    analyze_results(args.input, args.output)
