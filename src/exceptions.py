from __future__ import annotations


class DefenseTradingError(Exception):
    """Base exception for domain-specific failures."""


class ScannerFetchError(DefenseTradingError):
    """Raised when a scanner cannot fetch source data."""


class ScannerParseError(DefenseTradingError):
    """Raised when raw payload cannot be parsed into Event objects."""


class ConfigurationError(DefenseTradingError):
    """Raised for invalid runtime configuration usage."""
