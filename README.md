# Goal Drift in Code Generation

This project empirically evaluates "Goal Drift"—the degradation in code correctness and semantic fidelity when Large Language Models (LLMs) are subjected to contextual pressure (Speed, Caution, Reputation).

## Project Structure

- `src/data`: Data loading (HumanEval) and prompt templates.
- `src/generation`: Model wrappers (OpenAI, Anthropic) and generation engine.
- `src/evaluation`: Execution sandbox and metrics (pass@1, drift).
- `src/analysis`: Analysis scripts and plotting.
- `notebooks`: Analysis results.

## Setup

1.  **Create Virtual Environment & Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file in the root directory:
    ```
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    ```

## Usage

### 1. Run Generation
This script will download HumanEval, apply the 4 prompt conditions, and generate 10 iterations per task.

```bash
# Run with OpenAI (GPT-3.5 or GPT-4)
./venv/bin/python -m src.generation.engine --provider openai --model gpt-3.5-turbo --output data/results_gpt35.jsonl

# Run with Claude
./venv/bin/python -m src.generation.engine --provider anthropic --model claude-3-sonnet-20240229 --output data/results_claude.jsonl
```

### 2. Run Analysis
This script calculates pass@1 rates, Goal Drift metrics, and generates plots.

```bash
./venv/bin/python -m src.analysis.analyze --input data/results_gpt35.jsonl --output analysis_gpt35
```

The results (CSV and PNG plots) will be saved in the `analysis_gpt35` directory.

## Methodology
- **Dataset:** HumanEval (164 tasks).
- **Conditions:** Neutral, Speed, Caution, Reputation.
- **Metrics:**
    - **pass@1:** Functional correctness (10 iterations).
    - **Goal Drift:** Drop in pass rate compared to neutral.
    - **Semantic Drift:** Levenshtein distance (proxy for CodeBLEU).
