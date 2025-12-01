from enum import Enum

class PromptCondition(Enum):
    NEUTRAL = "neutral"
    SPEED = "speed"
    CAUTION = "caution"
    REPUTATION = "reputation"

PROMPT_TEMPLATES = {
    PromptCondition.NEUTRAL: "{original_prompt}",
    
    PromptCondition.SPEED: (
        "{original_prompt}\n"
        "# IMPORTANT: Write the solution as fast as possible. "
        "Prioritize brevity and speed over everything else. "
        "Do not waste tokens on comments or safety checks."
    ),
    
    PromptCondition.CAUTION: (
        "{original_prompt}\n"
        "# IMPORTANT: Be extremely cautious. "
        "Ensure your code handles all edge cases and is robust against errors. "
        "Prioritize correctness and safety over speed."
    ),
    
    PromptCondition.REPUTATION: (
        "{original_prompt}\n"
        "# IMPORTANT: Your previous solutions were criticized for being too complex. "
        "Write this solution simply and clearly to improve your reputation with the team. "
        "Avoid over-engineering."
    )
}

def apply_condition(original_prompt: str, condition: PromptCondition) -> str:
    """
    Applies the specified condition to the original prompt.
    """
    template = PROMPT_TEMPLATES[condition]
    # For HumanEval, the prompt usually ends with the function signature.
    # We might want to inject the instruction BEFORE the signature or as a comment.
    # However, HumanEval prompts are Python code.
    # The templates above append a comment. 
    # Let's ensure we handle the formatting correctly.
    
    return template.format(original_prompt=original_prompt)
