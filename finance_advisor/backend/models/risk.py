# backend/models/risk.py

from pydantic import BaseModel
from typing import Dict


class RiskProfileRequest(BaseModel):
    session_id: str
    age: int
    income_stability: str            # "low" | "medium" | "high"
    liquidity_needs: str             # "low" | "medium" | "high"
    investment_knowledge: str        # "low" | "medium" | "high"
    answers: Dict[str, int]          # Questionnaire scores (MCQs)


class RiskProfileResponse(BaseModel):
    risk_category: str               # "conservative", "moderate", "aggressive"
    score: int                       # Overall computed score
    explanation: str                 # Friendly explanation returned to UI
