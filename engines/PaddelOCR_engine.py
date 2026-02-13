"""
Concrete implementation of the OCREngine interface using PaddleOCR
instead of Tesseract, optimized for Vietnamese financial documents.
"""

from __future__ import annotations

from typing import Optional, List, Any

import numpy as np
from paddleocr import PaddleOCR  # type: ignore
from PIL import Image

from config import Config
from core.exceptions import OCREngineError
from core.interfaces import OCREngine


class PaddleOCREngine(OCREngine):
    """
    PaddleOCR-based engine (Vietnamese-optimized).
    
    This engine uses PaddleOCR instead of Tesseract for better accuracy
    with Vietnamese text, especially in financial reports.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
            self.config: Config = config or Config()

            # Logic chọn ngôn ngữ
            lang = "vi"
            if self.config.ocr_lang and "en" in self.config.ocr_lang.lower() and "vi" not in self.config.ocr_lang.lower():
                lang = "en"

            try:
                # --- CẤU HÌNH FIX LỖI CRASH TRÊN WINDOWS ---
                self._ocr = PaddleOCR(
                    lang=lang,
                    
                    # 1. Tắt tính năng xoay chiều (nguyên nhân chính tải thêm model UVDoc gây nặng máy)
                    use_angle_cls=False,  
                    
                    # 2. Tắt bộ tăng tốc MKLDNN (nguyên nhân gây crash "terminated abruptly" với model v5)
                    enable_mkldnn=False, 
                    
                
                )
            except Exception as exc:
                message = (
                    "Không khởi tạo được PaddleOCR. "
                    f"Lỗi gốc: {exc}"
                )
                raise OCREngineError(message) from exc

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Convert PIL Image to RGB (PaddleOCR requirement).
        """
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    def ocr_page(self, image: Image.Image) -> str:
        """
        Chạy OCR 1 trang.
        """
        # 1. Chuẩn bị ảnh
        img = self.preprocess_image(image)
        np_img = np.array(img)
        
        # PaddleOCR dùng thư viện OpenCV (BGR) nên đôi khi cần đảo channel
        # Tuy nhiên bản mới nhất thường tự xử lý, nhưng convert sang BGR là chuẩn nhất
        # np_img = np_img[:, :, ::-1] 

        try:
            # 2. Gọi PaddleOCR
            # result structure: [ [ [box], [text, score] ], ... ]
            result = self._ocr.ocr(np_img)
        except Exception as exc:
            raise OCREngineError(f"Lỗi khi chạy PaddleOCR: {exc}") from exc

        # --- FIX BUG: Xử lý trường hợp không tìm thấy chữ ---
        if not result or result[0] is None:
            return ""

        # 3. Trích xuất text
        # Paddle trả về list of lists (do hỗ trợ nhiều vùng ảnh), thường ta lấy result[0]
        # Nhưng với bản mới, result chính là list các line.
        
        lines: List[str] = []
        
        # Kiểm tra cấu trúc trả về để loop cho đúng (PaddleOCR output hơi lộn xộn tùy version)
        # Cách an toàn nhất: flatten list
        ocr_result = result[0] if (len(result) > 0 and isinstance(result[0], list)) else result
        
        if ocr_result:
            for line in ocr_result:
                # line structure: [box_coords, (text_content, confidence)]
                if line and len(line) >= 2:
                    text_content = line[1][0]
                    lines.append(text_content)

        # Nối các dòng lại. 
        # Lưu ý: Với báo cáo tài chính, nối bằng "\n" sẽ làm mất cấu trúc bảng (table).
        # Nhưng ở bước này ta cứ lấy raw text trước.
        return "\n".join(lines).strip()