# backend/models/currency.py

from pydantic import BaseModel


class CurrencyConversionResponse(BaseModel):
    from_currency: str          # Base currency (e.g., USD)
    to_currency: str            # Target currency (e.g., INR)
    rate: float                 # Current FX rate
    converted_amount: float     # Result after conversion
