"""
Worker functions for multiprocessing.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image

from config import Config
from core.exceptions import OCREngineError

# --- SỬA IMPORT TẠI ĐÂY ---
# Đã đổi tên file từ PaddelOCR_engine -> paddle_engine
try:
    from engines.paddle_engine import PaddleOCREngine
except ImportError:
    # Fallback nếu bạn chưa đổi tên file
    from engines.PaddelOCR_engine import PaddleOCREngine 

# ... (Phần còn lại của file worker.py giữ nguyên như cũ vì logic Singleton đã đúng) ...
_engine_instance: Optional[PaddleOCREngine] = None
_engine_config: Optional[Config] = None

def _get_global_engine(ocr_lang: str) -> PaddleOCREngine:
    global _engine_instance, _engine_config
    if _engine_instance is None or (_engine_config and _engine_config.ocr_lang != ocr_lang):
        # print(f"Init PaddleOCR in worker (PID: {os.getpid()})...") 
        config = Config(ocr_lang=ocr_lang)
        _engine_config = config
        _engine_instance = PaddleOCREngine(config=config)
    return _engine_instance

def process_single_page(image_array: np.ndarray, ocr_lang: str) -> str:
    try:
        engine = _get_global_engine(ocr_lang)
        image = Image.fromarray(image_array)
        return engine.ocr_page(image)
    except Exception as exc:
        raise OCREngineError(f"Worker Error: {exc}") from exc