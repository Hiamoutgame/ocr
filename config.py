"""
Application-wide configuration for the OCR financial report project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class Config:
    """
    Central configuration object.

    Paths are Windows-oriented by default but can be customized
    via CLI flags or environment variables.
    """

    # Path to tesseract.exe on Windows. Set to None if already in PATH.
    tesseract_cmd: Optional[str] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Path to Poppler binaries for pdf2image on Windows. Set to None if in PATH.
    poppler_path: Optional[str] = (
        r"C:\Program Files\poppler-25.12.0\Library\bin"
    )

    # DPI when converting PDF pages to images.
    pdf_dpi: int = 300

    # OCR language configuration (Vietnamese + English).
    ocr_lang: str = "vie+eng"

    # Page segmentation mode (PSM) and OCR engine mode (OEM).
    ocr_psm: int = 6
    ocr_oem: int = 3

    # Whether to use Otsu thresholding; otherwise, adaptive thresholding is used.
    use_otsu: bool = True

