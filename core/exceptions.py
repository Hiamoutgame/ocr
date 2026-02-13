"""
Custom exception classes for the OCR financial report pipeline.
"""

from __future__ import annotations


class OCRBaseError(Exception):
    """Base class for all custom OCR-related errors."""


class DocumentNotFoundError(OCRBaseError):
    """Raised when an input document cannot be found on disk."""


class DocumentTypeNotSupportedError(OCRBaseError):
    """Raised when the input document type/extension is not supported."""


class OCREngineError(OCRBaseError):
    """Raised for low-level OCR engine errors (Tesseract, etc.)."""


class ParsingError(OCRBaseError):
    """Raised when post-processing/parsing the OCR text fails."""

