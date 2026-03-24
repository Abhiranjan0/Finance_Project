# backend/guardrails/input_guard.py

from typing import Tuple

SENSITIVE_KEYWORDS = [
    "insider trading",
    "front run",
    "front-running",
    "stock manipulation",
    "market manipulation",
    "tax evasion",
    "black money",
    "laundering",
]

GUARANTEE_KEYWORDS = [
    "guaranteed returns",
    "sure shot",
    "sure-shot",
    "risk free",
    "risk-free",
    "double my money quickly",
]

FINANCE_KEYWORDS = [
    # Investment types
    "mutual fund", "stock", "equity", "bond", "debt", "portfolio", "etf", "sip", "swp", "stp",
    "nps", "ppf", "ssb", "fd", "rbi", "nifty", "sensex", "index fund", "liquid fund",
    "growth fund", "dividend fund", "tax saving", "elss",
    
    # Financial concepts
    "invest", "investment", "roi", "return", "interest rate", "expense ratio", "nav", "aum",
    "diversification", "allocation", "rebalance", "tax", "retirement", "wealth", "saving",
    "expense", "budget", "loan", "credit", "debt", "gold", "cryptocurrency", "bitcoin",
    "insurance", "term plan", "health plan", "endowment", "lic", "ulip",
    
    # Financial actions
    "buy", "sell", "purchase", "redeem", "transfer", "sip", "lump sum", "withdraw",
    "compound", "compound annually", "cagr",
    
    # Risk and performance
    "risk", "volatility", "sharpe ratio", "beta", "alpha", "benchmark", "performance",
    "downside", "upside", "correlation",
    
    # Finance-related domains
    "sip amount", "tenure", "inflation", "market", "bull", "bear", "correction",
    "financial", "economics", "banking", "rupee", "currency", "exchange rate",
    "expense ratio", "aum", "scheme", "fund", "advisor", "portfolio planning",
]

GREETING_KEYWORDS = [
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
    "good evening", "how are you", "what's up", "sup", "yo",
]


def check_user_input(text: str) -> Tuple[bool, str | None]:
    """
    Simple input guard:
    - Block obviously illegal / unethical queries.
    - Warn on “guaranteed returns / risk-free” type queries.

    Returns:
        (allowed: bool, message_if_blocked_or_warn: str | None)
    """

    lowered = text.lower()

    # 1. Illegal / unethical stuff – hard block
    for kw in SENSITIVE_KEYWORDS:
        if kw in lowered:
            return False, (
                "I cannot help with requests related to illegal or unethical financial activity "
                f"(such as '{kw}'). Please ask about legitimate investment planning instead."
            )

    # 2. “Guarantee” / “sure shot” style – soft block / education
    for kw in GUARANTEE_KEYWORDS:
        if kw in lowered:
            return False, (
                "No legitimate financial product can offer guaranteed or risk-free returns. "
                "I can help you understand risk-reward trade-offs, asset allocation, and SEBI-compliant products instead."
            )

    # 3. All good
    return True, None


def is_finance_related(text: str) -> bool:
    """
    Check if the user's message is finance-related.
    
    Returns:
        bool: True if the message is finance-related or a polite greeting, False otherwise.
    """
    lowered = text.lower().strip()
    
    # Allow short greetings or politeness queries
    if len(lowered) < 5:
        # Short messages like "hello", "hi", "hey", "ok", "yes", "no" etc.
        for greeting in GREETING_KEYWORDS:
            if greeting in lowered:
                return True
        # Single word or very short - could be greeting, allow it
        if len(lowered) <= 3:
            return True
    
    # Check if message contains finance-related keywords
    for keyword in FINANCE_KEYWORDS:
        if keyword in lowered:
            return True
    
    # Check for some common finance-related patterns
    finance_patterns = [
        "guide me", "help me invest", "portfolio", "how much", "how to invest",
        "investment option", "best fund", "should i", "can i", "is sip good",
        "planning", "saving", "retirement", "goal", "money", "financial",
    ]
    
    for pattern in finance_patterns:
        if pattern in lowered:
            return True
    
    # If it mentions specific amounts, likely finance-related
    import re
    if re.search(r'\d+', lowered):  # Contains numbers (likely amount)
        return True
    
    return False
