"""
PDF loader implementation using pdf2image.
"""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from config import Config
from core.exceptions import (
    DocumentNotFoundError,
    DocumentTypeNotSupportedError,
)
from core.interfaces import DocumentLoader


class PdfLoader(DocumentLoader):
    """Load a multi-page PDF document into a list of images."""

    def __init__(self, config: Config | None = None) -> None:
        self.config: Config = config or Config()

    def load(self, file_path: str | Path) -> list[Image.Image]:
        path = Path(file_path)
        if not path.exists():
            raise DocumentNotFoundError(f"Không tìm thấy file: {path}")

        if path.suffix.lower() != ".pdf":
            raise DocumentTypeNotSupportedError(
                f"Định dạng không hỗ trợ cho PdfLoader: {path.suffix.lower()}"
            )

        kwargs: dict[str, object] = {"dpi": self.config.pdf_dpi}
        if self.config.poppler_path:
            kwargs["poppler_path"] = self.config.poppler_path

        pages = convert_from_path(str(path), **kwargs)
        return list(pages)

