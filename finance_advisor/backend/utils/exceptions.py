# backend/utils/exceptions.py

class AdvisorException(Exception):
    """Base class for all advisor-specific exceptions."""
    pass


class MissingDataException(AdvisorException):
    """Raised when required user info is missing in memory."""
    pass


class ExternalAPIException(AdvisorException):
    """Raised when external API (NAV/Fx) fails."""
    pass


class PortfolioConstructionException(AdvisorException):
    """Raised when portfolio construction fails."""
    pass


class SimulationException(AdvisorException):
    """Raised when simulation encounters invalid data."""
    pass
