import httpx
from datetime import date
from typing import Dict, Any, Optional
from difflib import get_close_matches


# ------------------------------
# CONFIG / CONSTANTS
# ------------------------------

# MFAPI.in - Free API for Indian Mutual Funds (No API key required)
BASE_URL = "https://api.mfapi.in/mf"
#API_KEY = None  # Not required for MFAPI.in (completely free)

FALLBACK_NAV = 100.0  # Default NAV if APIs fail


# ------------------------------
# Helper: Build Unified Output
# ------------------------------
def _build_output(symbol: str, date_str: str, nav: float,
                  fund_type: str = "unknown", risk_level: str = "unknown",
                  **kwargs) -> Dict[str, Any]:
    """Build standardized output dictionary"""
    result = {
        "symbol": symbol,
        "date": date_str,
        "nav": float(nav) if nav else None,
        "fund_type": fund_type,
        "risk_level": risk_level
    }
    # Add any additional fields
    result.update(kwargs)
    return result


# ------------------------------
# Helper: Determine Fund Type
# ------------------------------
def _determine_fund_type(scheme_name: str) -> str:
    """Determine fund type based on scheme name"""
    scheme_lower = scheme_name.lower()
    
    if 'equity' in scheme_lower or 'stock' in scheme_lower:
        return 'equity'
    elif 'debt' in scheme_lower or 'bond' in scheme_lower or 'income' in scheme_lower:
        return 'debt'
    elif 'hybrid' in scheme_lower or 'balanced' in scheme_lower:
        return 'hybrid'
    elif 'liquid' in scheme_lower or 'money market' in scheme_lower:
        return 'liquid'
    elif 'gilt' in scheme_lower:
        return 'gilt'
    elif 'elss' in scheme_lower or 'tax' in scheme_lower:
        return 'elss'
    elif 'index' in scheme_lower:
        return 'index'
    elif 'fof' in scheme_lower or 'fund of fund' in scheme_lower:
        return 'fund_of_funds'
    else:
        return 'other'


# ------------------------------
# Helper: Determine Risk Level
# ------------------------------
def _determine_risk_level(fund_type: str, scheme_name: str) -> str:
    """Determine risk level based on fund type and name"""
    scheme_lower = scheme_name.lower()
    
    # High risk
    if fund_type == 'equity' or 'small cap' in scheme_lower or 'mid cap' in scheme_lower:
        return 'high'
    
    # Moderately High risk
    if 'large cap' in scheme_lower or 'multi cap' in scheme_lower or fund_type == 'hybrid':
        return 'moderately_high'
    
    # Low to Moderate risk
    if fund_type == 'debt' or 'short term' in scheme_lower:
        return 'low_to_moderate'
    
    # Very Low risk
    if fund_type == 'liquid' or 'ultra short' in scheme_lower or fund_type == 'gilt':
        return 'very_low'
    
    return 'moderate'


# ------------------------------
# Main Function: NAV Fetch
# ------------------------------
def fetch_nav_data(symbol: str, date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches NAV data for an Indian mutual fund.
    Returns a clean dict for NAVResponse model.
    
    Args:
        symbol: Mutual fund name or scheme code (e.g., "HDFC Equity Fund" or "119551")
        date_str: Optional date (not used as API returns latest NAV)
    
    Returns:
        Dict with keys: symbol, date, nav, fund_type, risk_level, scheme_code, 
                       fund_house, scheme_category, scheme_type
    """
    
    if not date_str:
        date_str = date.today().isoformat()

    # -----------------------------------
    # Step 1: Search for the fund
    # -----------------------------------
    try:
        with httpx.Client(timeout=15.0) as client:
            # Get all funds list
            resp = client.get(BASE_URL)
            resp.raise_for_status()
            all_funds = resp.json()
            
            # Determine if symbol is scheme code or name
            if symbol.isdigit():
                # Direct scheme code lookup
                scheme_code = symbol
                matching_fund = next((f for f in all_funds if f['schemeCode'] == symbol), None)
                if not matching_fund:
                    return _build_output(
                        symbol=symbol,
                        date_str=date_str,
                        nav=FALLBACK_NAV,
                        fund_type="unknown",
                        risk_level="unknown",
                        error=f"No mutual fund found with scheme code '{symbol}'"
                    )
            else:
                # Fuzzy search by name
                fund_names_map = {fund['schemeName']: fund['schemeCode'] for fund in all_funds}
                matches = get_close_matches(symbol, fund_names_map.keys(), n=1, cutoff=0.4)
                
                if not matches:
                    return _build_output(
                        symbol=symbol,
                        date_str=date_str,
                        nav=FALLBACK_NAV,
                        fund_type="unknown",
                        risk_level="unknown",
                        error=f"No mutual fund found matching '{symbol}'. Try using scheme code or more specific name."
                    )
                
                scheme_code = fund_names_map[matches[0]]
            
            # -----------------------------------
            # Step 2: Fetch fund details
            # -----------------------------------
            detail_url = f"{BASE_URL}/{scheme_code}"
            detail_resp = client.get(detail_url)
            detail_resp.raise_for_status()
            details = detail_resp.json()
            
            # Extract metadata
            scheme_name = details['meta']['scheme_name']
            fund_house = details['meta']['fund_house']
            scheme_category = details['meta'].get('scheme_category', 'N/A')
            scheme_type = details['meta'].get('scheme_type', 'N/A')
            
            # Extract latest NAV
            if details['data'] and len(details['data']) > 0:
                nav_value = float(details['data'][0]['nav'])
                nav_date = details['data'][0]['date']
            else:
                nav_value = FALLBACK_NAV
                nav_date = date_str
            
            # Determine fund type and risk
            fund_type = _determine_fund_type(scheme_name)
            risk_level = _determine_risk_level(fund_type, scheme_name)
            
            # Build result
            result = _build_output(
                symbol=scheme_name,
                date_str=nav_date,
                nav=nav_value,
                fund_type=fund_type,
                risk_level=risk_level,
                scheme_code=scheme_code,
                fund_house=fund_house,
                scheme_category=scheme_category,
                scheme_type=scheme_type
            )
            
            return result
            
    except httpx.HTTPStatusError as e:
        # HTTP error from API
        return _build_output(
            symbol=symbol,
            date_str=date_str,
            nav=FALLBACK_NAV,
            fund_type="unknown",
            risk_level="unknown",
            error=f"API error: {e.response.status_code}"
        )
    except httpx.RequestError as e:
        # Network/connection error
        return _build_output(
            symbol=symbol,
            date_str=date_str,
            nav=FALLBACK_NAV,
            fund_type="unknown",
            risk_level="unknown",
            error=f"Network error: {str(e)}"
        )
    except Exception as e:
        # -----------------------------------
        # Step 3: Fallback NAV
        # -----------------------------------
        return _build_output(
            symbol=symbol,
            date_str=date_str,
            nav=FALLBACK_NAV,
            fund_type="unknown",
            risk_level="unknown",
            error=f"Unexpected error : {str(e)}"
        )