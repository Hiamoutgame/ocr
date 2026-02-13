"""
Abstract interfaces for the OCR financial report pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


class DocumentLoader(ABC):
    """Abstract interface for loading documents (PDF, images, etc.)."""

    @abstractmethod
    def load(self, file_path: str | Path) -> list[Image.Image]:
        """Load the input document into a list of PIL images."""


class OCREngine(ABC):
    """Abstract interface for OCR engines."""

    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply pre-processing to a single image before OCR."""

    @abstractmethod
    def ocr_page(self, image: Image.Image) -> str:
        """Run OCR on a single (optionally preprocessed) image."""

    def ocr_pages(self, images: Sequence[Image.Image]) -> list[str]:
        """
        Optional helper to OCR multiple pages sequentially.

        Concurrency (e.g. ProcessPoolExecutor) should be handled
        at a higher level (e.g. in main.py or a service layer).
        """
        return [self.ocr_page(image) for image in images]


class DataParser(ABC):
    """Abstract interface for post-processing raw OCR text."""

    @abstractmethod
    def parse(self, text: str) -> dict[str, Any]:
        """
        Transform raw OCR text into a structured Python object
        (e.g. JSON-like dict) suitable for persistence.
        """

