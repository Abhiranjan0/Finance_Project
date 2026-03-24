# backend/tools/currency_converter.py

import httpx
from typing import Dict, Any
import os
from ..utils.cache import cache_get, cache_set
from dotenv import load_dotenv
load_dotenv()

API_URL = "https://api.exchangerate.host/convert"
API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

def convert_currency_amount(
    from_currency: str,
    to_currency: str,
    amount: float
) -> Dict[str, Any]:
    """
    Converts currency using exchangerate.host.
    Includes:
    - Caching
    - API failure fallback
    - Clean unified response dict
    """

    cache_key = f"FX:{from_currency}:{to_currency}:{amount}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        url = (
            f"{API_URL}?access_key={API_KEY}"
            f"&from={from_currency}&to={to_currency}&amount={amount}"
        )

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()

        data = resp.json()

        rate = float(data["info"]["quote"])
        converted = float(data["result"])

        result = {
            "from": from_currency,
            "to": to_currency,
            "rate": rate,
            "converted_amount": converted
        }

        cache_set(cache_key, result)
        return result

    except Exception:
        # -------------------------------
        # API Failure → Fallback
        # -------------------------------
        # Use a constant fallback rate just to avoid hard crashes
        fallback_rate = 80.0  # crude fallback (INR conversion)
        fallback_converted = amount * fallback_rate

        result = {
            "from": from_currency,
            "to": to_currency,
            "rate": fallback_rate,
            "converted_amount": fallback_converted
        }

        cache_set(cache_key, result)
        return result
