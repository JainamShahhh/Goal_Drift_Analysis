import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_mining(input_file: str, output_dir: str):
    # Check for stats file first
    stats_file = input_file.replace("mining_results.json", "mining_stats.json")
    if os.path.exists(stats_file):
        print(f"Loading mining stats from {stats_file}...")
        with open(stats_file, "r") as f:
            counts = json.load(f)
    else:
        print(f"Loading mining results from {input_file}...")
        with open(input_file, "r") as f:
            data = json.load(f)
        counts = {category: len(items) for category, items in data.items()}
    
    # Sort counts descending
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    print("Mining Counts:", sorted_counts)
    
    # Calculate Total Scanned
    total_scanned = sum(sorted_counts.values())
    print(f"Total Discussions Scanned: {total_scanned}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Bar Chart (Top 10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(sorted_counts.values()), y=list(sorted_counts.keys()), palette="viridis")
    plt.title(f"Prevalence of Developer Constraints (N={total_scanned:,} Discussions)")
    plt.xlabel("Number of Issues/PRs Found")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_0_mining_counts.png")
    plt.close()
    
    # Plot 2: Pie Chart (Top 5 for readability)
    top_5 = dict(list(sorted_counts.items())[:5])
    plt.figure(figsize=(8, 8))
    plt.pie(top_5.values(), labels=top_5.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Distribution of Top 5 Constraints")
    plt.savefig(f"{output_dir}/plot_0_mining_distribution.png")
    plt.close()
    
    print(f"Mining plots saved to {output_dir}")

if __name__ == "__main__":
    visualize_mining("data/mining_results.json", "analysis_combined")
