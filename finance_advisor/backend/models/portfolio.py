from pydantic import BaseModel
from typing import Dict

class PortfolioResponse(BaseModel):
    allocation: Dict[str, float]
    explanation: str
