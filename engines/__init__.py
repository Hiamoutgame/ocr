"""
Engine package exports.

Currently we expose the PaddleOCR-based implementation via the
`TesseractEngine` name to keep backward compatibility with existing
imports in the project.
"""

from __future__ import annotations

from .PaddelOCR_engine import TesseractEngine  # noqa: F401

