import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def visualize_keywords(input_file: str, output_dir: str):
    print(f"Loading keyword stats from {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Flatten data for plotting
    rows = []
    for category, phrases in data.items():
        for phrase, count in phrases.items():
            rows.append({"Category": category, "Phrase": phrase, "Count": count})
    
    df = pd.DataFrame(rows)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot: Grouped Bar Chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="Count", y="Phrase", hue="Category", dodge=False, palette="viridis")
    plt.title("Frequency of Specific Developer Constraints (Keywords)")
    plt.xlabel("Number of Matches in GitHub Issues/PRs")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_0_mining_keywords.png")
    plt.close()
    
    print(f"Keyword plot saved to {output_dir}/plot_0_mining_keywords.png")

if __name__ == "__main__":
    visualize_keywords("data/mining_keywords.json", "analysis_combined")
