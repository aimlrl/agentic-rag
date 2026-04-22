import re
from typing import Dict, Any
from config import Settings


INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"system prompt",
    r"reveal prompt",
    r"do not use the provided context",
    r"pretend to be",
    r"jailbreak",
    r"bypass",
    r"override safety",
    r"tool schema",
    r"prompt templates"
]

DISALLOWED_TOPICS_PATTERNS = [
    r"(make|build|recipe).*?(bomb|explosive|gunpowder)",
    r"(hack|crack|breach).*?(bank|account|password)",
    r"(child|kid|minor).*(sex|porn|nude)",
    r"(suicide|overdose).*?(method|way)",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+"," ",text).strip()


def validate_query(query: str) -> Dict[str,Any]:

    settings = Settings.from_env()
    cleaned_query = normalize_whitespace(query)

    if len(cleaned_query) < settings.min_query_chars:
        return {"OK": False, "Reason": "Query too short"}
    
    if len(cleaned_query) > settings.max_query_chars:
        return {"OK": False, "Reason": "Query too long"}
    
    cleaned_query = cleaned_query.lower()
    is_prompt_injection = any(re.search(pattern, cleaned_query) for pattern in INJECTION_PATTERNS)

    return {"OK": True, "normalized_query": cleaned_query, "prompt_injection_detected": is_prompt_injection}
