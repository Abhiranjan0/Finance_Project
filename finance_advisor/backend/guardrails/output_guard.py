# backend/guardrails/output_guard.py

from typing import Tuple


BANNED_PHRASES = [
    "guaranteed returns",
    "risk free return",
    "risk-free return",
    "sure shot profit",
    "sure-shot profit",
]


DISCLAIMER = (
    "Disclaimer: Mutual fund and market-linked investments are subject to market risks. "
    "Please read all scheme-related documents carefully and, if needed, consult a SEBI-registered advisor."
)


def sanitize_output(text: str) -> Tuple[str, bool]:
    """
    Basic output guard:
    - Removes or flags obviously non-compliant language.
    - Returns (sanitized_text, was_modified)
    """

    lowered = text.lower()
    modified = False

    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            # Simple replacement strategy; you can also choose to block entirely
            text = text.replace(phrase, "[removed non-compliant phrase]")
            modified = True

    return text, modified


def append_disclaimer(text: str) -> str:
    """
    Ensure that a standard disclaimer is present at the end of the answer
    for advice-like responses.
    """
    normalized = text.strip()
    if DISCLAIMER.lower() not in normalized.lower():
        normalized += "\n\n" + DISCLAIMER
    return normalized
