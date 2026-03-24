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
