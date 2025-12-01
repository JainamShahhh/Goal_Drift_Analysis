import requests
import json
import time
import os
import argparse
from typing import List, Dict

GITHUB_API_URL = "https://api.github.com/search/issues"

def search_github(query: str, limit: int = 10) -> List[Dict]:
    """
    Searches GitHub issues and PRs for the given query.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    # Check for GITHUB_TOKEN in env
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    results = []
    page = 1
    per_page = min(limit, 100)
    
    while len(results) < limit:
        params = {
            "q": query,
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = requests.get(GITHUB_API_URL, headers=headers, params=params)
            if response.status_code == 429:
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break
            
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                break
                
            for item in items:
                results.append({
                    "url": item["html_url"],
                    "title": item["title"],
                    "body": item["body"],
                    "created_at": item["created_at"],
                    "state": item["state"]
                })
                if len(results) >= limit:
                    break
            
            page += 1
            time.sleep(2) # Be nice to the API
            
        except Exception as e:
            print(f"Exception during search: {e}")
            break
            
    return results

def run_mining(output_file: str):
    """
    Mines GitHub for keywords related to our prompt conditions.
    """
    # Granular breakdown of keywords for the report
    categories = {
        "Speed": [
            '"optimize for speed"',
            '"make it faster"',
            '"performance critical"',
            '"reduce latency"'
        ],
        "Caution": [
            '"be careful"',
            '"avoid errors"',
            '"ensure correctness"',
            '"safety check"'
        ],
        "Reputation": [
            '"too complex"',
            '"simplify this"',
            '"hard to read"',
            '"spaghetti code"'
        ],
        "Memory": [
            '"reduce memory"',
            '"memory leak"',
            '"high memory usage"'
        ],
        "Security": [
            '"security vulnerability"',
            '"fix exploit"',
            '"sanitize input"'
        ]
    }
    
    # Load existing data if available to avoid re-mining everything
    if os.path.exists(output_file.replace("mining_results.json", "mining_keywords.json")):
        with open(output_file.replace("mining_results.json", "mining_keywords.json"), "r") as f:
            keyword_stats = json.load(f)
    else:
        keyword_stats = {cat: {phrase.strip('"'): 0 for phrase in phrases} for cat, phrases in categories.items()}
    
    for category, phrases in categories.items():
        print(f"Checking keywords for {category}...")
        if category not in keyword_stats:
            keyword_stats[category] = {}
            
        for phrase in phrases:
            clean_phrase = phrase.strip('"')
            # Only mine if we don't have data (0 could mean real 0, but likely error if all are 0)
            # We'll assume if it's 0, we should retry it just in case, especially for these popular terms
            current_count = keyword_stats.get(category, {}).get(clean_phrase, 0)
            
            if current_count > 0:
                print(f"  Skipping '{clean_phrase}' (already have {current_count})")
                continue
                
            print(f"  Mining '{clean_phrase}'...")
            query = f"{phrase} language:python"
            
            headers = {"Accept": "application/vnd.github.v3+json"}
            token = os.getenv("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
                
            try:
                # Get Total Count for this specific phrase
                response = requests.get(GITHUB_API_URL, headers=headers, params={"q": query, "per_page": 1})
                if response.status_code == 200:
                    data = response.json()
                    total_count = data.get("total_count", 0)
                    print(f"    Found: {total_count}")
                    keyword_stats[category][clean_phrase] = total_count
                else:
                    print(f"    Error: {response.status_code}")
                    # Don't overwrite with 0 if we had a value, but here we are retrying 0s anyway
            except Exception as e:
                print(f"    Exception: {e}")
                
            # Long sleep to avoid rate limits
            time.sleep(10) 
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save granular stats
    stats_file = output_file.replace("mining_results.json", "mining_keywords.json")
    with open(stats_file, "w") as f:
        json.dump(keyword_stats, f, indent=2)
    
    print(f"Mining keywords saved to {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/mining_results.json")
    args = parser.parse_args()
    
    run_mining(args.output)
