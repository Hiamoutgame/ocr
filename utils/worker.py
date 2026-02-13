"""
Worker functions for multiprocessing (ProcessPoolExecutor).

IMPORTANT: All functions used by ProcessPoolExecutor must be defined
at module top-level to remain picklable, especially on Windows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from config import Config
from engines.PaddelOCR_engine import TesseractEngine


def process_single_page(
    image_array: np.ndarray,
    tesseract_cmd: Optional[str],
    lang: str,
    psm: int,
    oem: int,
    use_otsu: bool,
) -> str:
    """
    Process a single page: preprocessing + OCR.

    This function is intended to be executed inside worker processes
    managed by ProcessPoolExecutor.
    """
    config = Config(
        tesseract_cmd=tesseract_cmd,
        ocr_lang=lang,
        ocr_psm=psm,
        ocr_oem=oem,
        use_otsu=use_otsu,
    )
    engine = TesseractEngine(config=config)

    # Convert numpy array back to PIL.Image for the engine.
    image = Image.fromarray(image_array)
    preprocessed = engine.preprocess_image(image)
    return engine.ocr_page(preprocessed)

