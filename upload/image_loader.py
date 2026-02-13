"""
Image loader implementation for single-image financial reports.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.exceptions import DocumentNotFoundError, DocumentTypeNotSupportedError
from core.interfaces import DocumentLoader


class ImageLoader(DocumentLoader):
    """Load a single image file (PNG, JPG, etc.) as a one-page document."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def load(self, file_path: str | Path) -> list[Image.Image]:
        path = Path(file_path)
        if not path.exists():
            raise DocumentNotFoundError(f"Không tìm thấy file: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentTypeNotSupportedError(f"Định dạng không hỗ trợ: {ext}")

        image = Image.open(path).convert("RGB")
        return [image]

