"""
Concrete implementation of the OCREngine interface using PaddleOCR
instead of Tesseract, optimized for Vietnamese financial documents.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from paddleocr import PaddleOCR  # type: ignore[import-untyped]
from PIL import Image

from config import Config
from core.exceptions import OCREngineError
from core.interfaces import OCREngine


class TesseractEngine(OCREngine):
    """
    PaddleOCR-based engine (Vietnamese-optimized).

    NOTE:
        Tên class được giữ nguyên (`TesseractEngine`) để tránh phải sửa
        toàn bộ các import hiện có trong project, nhưng implementation
        bên trong đã chuyển sang dùng PaddleOCR.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config: Config = config or Config()

        # Map cấu hình ngôn ngữ cũ (ví dụ: "vie+eng") sang mã ngôn ngữ PaddleOCR.
        lang = "vi"
        if "en" in self.config.ocr_lang.lower() and "vi" not in self.config.ocr_lang.lower():
            lang = "en"

        try:
            # `use_gpu=False` mặc định cho môi trường không GPU.
            # Nếu bạn có GPU, có thể set lại ở đây.
            self._ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=False,
                show_log=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise OCREngineError(
                "Không khởi tạo được PaddleOCR. "
                "Hãy đảm bảo đã cài paddlepaddle và paddleocr."
            ) from exc

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Tiền xử lý cơ bản cho PaddleOCR.

        PaddleOCR đã có pipeline riêng cho detection/recognition, nên
        ở đây ta chỉ cần đảm bảo ảnh ở dạng RGB. Nếu bạn muốn có
        các bước enhance (denoise, sharpen, v.v.) thì có thể bổ sung
        sau trong hàm này.
        """
        return image.convert("RGB")

    def ocr_page(self, image: Image.Image) -> str:
        """
        Chạy OCR 1 trang bằng PaddleOCR.

        Trả về: text ghép từ tất cả các dòng nhận dạng được.
        """
        # PaddleOCR nhận numpy array BGR hoặc đường dẫn file.
        img = self.preprocess_image(image)
        np_img = np.array(img)
        # RGB -> BGR
        np_img = np_img[:, :, ::-1]

        try:
            result = self._ocr.ocr(np_img, cls=True)
        except Exception as exc:  # noqa: BLE001
            raise OCREngineError(f"Lỗi khi chạy PaddleOCR: {exc}") from exc

        lines: list[str] = []
        # result: List[List[ [box, (text, score)] ]]
        for line in result:
            for _box, (text, _score) in line:
                lines.append(text)

        return "\n".join(lines).strip()
