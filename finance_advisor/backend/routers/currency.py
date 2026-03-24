# backend/routers/currency.py

from fastapi import APIRouter, Query, HTTPException

from ..models.currency import CurrencyConversionResponse
from ..tools.currency_convertor import convert_currency_amount

router = APIRouter(prefix="/convert_currency", tags=["currency"])


@router.get("", response_model=CurrencyConversionResponse)
def convert_currency(
    from_currency: str = Query(..., alias="from", description="Base currency code (e.g., USD)"),
    to_currency: str = Query(..., alias="to", description="Target currency code (e.g., INR)"),
    amount: float = Query(..., description="Amount to convert")
):
    """
    Converts any currency amount using exchangerate.host API or cached data.
    """
    try:
        result = convert_currency_amount(
            from_currency=from_currency.upper(),
            to_currency=to_currency.upper(),
            amount=amount,
        )

        return CurrencyConversionResponse(
            from_currency=result["from"],
            to_currency=result["to"],
            rate=result["rate"],
            converted_amount=result["converted_amount"],
        )

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
