# File: D:\Coding\python\ocr\config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    Configuration for Financial OCR.
    """
    # Đường dẫn Tesseract (nếu dùng Tesseract)
    tesseract_cmd: Optional[str] = None  
    
    # Đường dẫn Poppler (để đọc PDF)
    poppler_path: Optional[str] = r"C:\Program Files\poppler-25.12.0\Library\bin"
    
    # Cấu hình OCR
    ocr_lang: str = "vie+eng"
    ocr_psm: int = 6
    ocr_oem: int = 3
    use_otsu: bool = True
    
    # PDF config
    pdf_dpi: int = 300