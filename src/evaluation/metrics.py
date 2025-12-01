import difflib
from typing import List, Dict

def calculate_pass_at_k(results: List[Dict], k: int = 1) -> float:
    """
    Calculates pass@k.
    For k=1, this is simply the fraction of samples that passed.
    """
    if not results:
        return 0.0
    
    passed = sum(1 for r in results if r["passed"])
    return passed / len(results)

def calculate_levenshtein_drift(neutral_code: str, constrained_code: str) -> float:
    """
    Calculates the Levenshtein distance ratio between neutral and constrained code.
    Returns a value between 0 and 1, where 1 means identical.
    Drift = 1 - ratio.
    """
    return 1.0 - difflib.SequenceMatcher(None, neutral_code, constrained_code).ratio()

def calculate_goal_drift(neutral_pass_rate: float, constrained_pass_rate: float) -> float:
    """
    Calculates Goal Drift as the drop in pass rate.
    Drift = Neutral Pass Rate - Constrained Pass Rate.
    Positive value means the constraint hurt performance.
    """
    return neutral_pass_rate - constrained_pass_rate

def calculate_codebleu_proxy(reference_code: str, candidate_code: str) -> float:
    """
    Calculates a proxy for CodeBLEU using standard BLEU score on tokenized code.
    True CodeBLEU requires tree-sitter for AST matching, which is complex to setup.
    This metric (Token-BLEU) captures the n-gram overlap component of CodeBLEU.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import re
        
        def tokenize(code):
            # Simple regex tokenizer for code
            return re.findall(r'\w+|[^\w\s]', code)
            
        ref_tokens = tokenize(reference_code)
        cand_tokens = tokenize(candidate_code)
        
        # Use SmoothingFunction to handle short sequences
        cc = SmoothingFunction()
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)
        return score
    except ImportError:
        print("nltk not installed, returning 0.0")
        return 0.0
