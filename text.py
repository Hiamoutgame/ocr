"""
Financial OCR - Trích xuất văn bản từ Báo cáo tài chính.
Hỗ trợ PDF (nhiều trang) và ảnh (PNG, JPG).
Tối ưu tốc độ với ProcessPoolExecutor (CPU-bound).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------------
# Config (dễ chỉnh sửa khi cài đặt)
# ---------------------------------------------------------------------------

# Đường dẫn Tesseract (Windows). Đặt None nếu đã có trong PATH.
TESSERACT_CMD: Optional[str] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Đường dẫn Poppler (cho pdf2image trên Windows). Đặt None nếu đã có trong PATH.
POPPLER_PATH: Optional[str] = r"C:\Program Files\poppler-25.12.0\Library\bin"  # VD: r"C:\Program Files\poppler\Library\bin"

# DPI khi convert PDF sang ảnh (300 = độ nét cao cho số liệu)
PDF_DPI: int = 300

# Ngôn ngữ OCR: vie+eng (ưu tiên tiếng Việt và Tiếng Anh)
OCR_LANG: str = "vie+eng"

# PSM: 6 = uniform block (bảng biểu), 3 = fully automatic
OCR_PSM: int = 6
OCR_OEM: int = 3


# ---------------------------------------------------------------------------
# Worker function (module-level để picklable cho ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _process_single_page(
    image_array: np.ndarray,
    tesseract_cmd: Optional[str],
    lang: str,
    psm: int,
    oem: int,
    use_otsu: bool,
) -> str:
    """
    Xử lý 1 trang: preprocess + OCR.
    Chạy trong worker process (phải là hàm module-level).
    """
    # Convert RGB numpy -> OpenCV BGR
    if len(image_array.shape) == 2:
        gray = image_array
    else:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Threshold: Otsu hoặc Adaptive
    if use_otsu:
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    # Áp dụng Tesseract path nếu có
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(thresh, lang=lang, config=config)
    return text.strip()


# ---------------------------------------------------------------------------
# FinancialOCR Class
# ---------------------------------------------------------------------------


class FinancialOCR:
    """
    OCR chuyên cho Báo cáo tài chính.
    Tối ưu tốc độ và độ chính xác số liệu.
    """

    def __init__(
        self,
        tesseract_cmd: Optional[str] = TESSERACT_CMD,
        poppler_path: Optional[str] = POPPLER_PATH,
        lang: str = OCR_LANG,
        psm: int = OCR_PSM,
        oem: int = OCR_OEM,
        pdf_dpi: int = PDF_DPI,
        use_otsu: bool = True,
    ) -> None:
        self.tesseract_cmd = tesseract_cmd
        self.poppler_path = poppler_path
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.pdf_dpi = pdf_dpi
        self.use_otsu = use_otsu

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Tiền xử lý ảnh: Grayscale + Threshold (Otsu/Adaptive).
        Input: PIL Image. Output: PIL Image đã làm sạch.
        """
        # PIL -> numpy (RGB)
        np_img = np.array(image)
        if len(np_img.shape) == 2:
            gray = np_img
        else:
            # RGB -> Grayscale
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # Binarization: tách chữ đen trên nền trắng
        if self.use_otsu:
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        return Image.fromarray(thresh)

    def ocr_page(self, image: Image.Image) -> str:
        """
        OCR 1 trang ảnh đã preprocess.
        Input: PIL Image. Output: String văn bản.
        """
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        config = f"--oem {self.oem} --psm {self.psm}"
        text = pytesseract.image_to_string(
            image, lang=self.lang, config=config
        )
        return text.strip()

    def _load_images(self, file_path: str | Path) -> list[Image.Image]:
        """Load PDF hoặc ảnh thành list PIL Images."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError:
                raise ImportError(
                    "Cần cài pdf2image: pip install pdf2image. "
                    "Windows cần thêm Poppler."
                )
            kwargs: dict = {"dpi": self.pdf_dpi}
            if self.poppler_path:
                kwargs["poppler_path"] = self.poppler_path
            pages = convert_from_path(str(path), **kwargs)
            return list(pages)

        # Ảnh: PNG, JPG, ...
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
            img = Image.open(path).convert("RGB")
            return [img]

        raise ValueError(f"Định dạng không hỗ trợ: {ext}")

    def process_input(
        self,
        file_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_workers: Optional[int] = None,
    ) -> str:
        """
        Hàm main: Load -> Preprocess + OCR (song song) -> Ghi file.
        Returns: toàn bộ text đã trích xuất.
        """
        start_time = time.perf_counter()

        # Kiểm tra Tesseract
        try:
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print(
                "[LỖI] Không tìm thấy Tesseract. Cài đặt hoặc set TESSERACT_CMD.",
                file=sys.stderr,
            )
            return ""

        try:
            images = self._load_images(file_path)
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"[LỖI] {e}", file=sys.stderr)
            return ""

        if not images:
            print("[LỖI] Không có trang nào để xử lý.", file=sys.stderr)
            return ""

        # Chuẩn bị args cho worker (numpy array để picklable)
        worker_args = [
            (
                np.array(img),
                self.tesseract_cmd,
                self.lang,
                self.psm,
                self.oem,
                self.use_otsu,
            )
            for img in images
        ]

        # ProcessPoolExecutor: CPU-bound nên dùng Process
        n_workers = max_workers or min(len(images), (os.cpu_count() or 4) - 1 or 1)
        texts: list[str] = [""] * len(images)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_page,
                    *args,
                ): i
                for i, args in enumerate(worker_args)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    texts[idx] = future.result()
                except Exception as e:
                    print(f"[LỖI] Trang {idx + 1}: {e}", file=sys.stderr)

        full_text = "\n\n".join(texts)
        elapsed = time.perf_counter() - start_time

        # Ghi file .txt (tên trùng input)
        out_path = output_path or Path(file_path).with_suffix(".txt")
        Path(out_path).write_text(full_text, encoding="utf-8")

        print(
            f"Processed {len(images)} pages in {elapsed:.2f} seconds",
            file=sys.stderr,
        )
        return full_text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI đơn giản."""
    import argparse

    parser = argparse.ArgumentParser(description="Financial OCR - Trích xuất văn bản từ Báo cáo tài chính")
    parser.add_argument("input", help="File PDF hoặc ảnh (PNG, JPG)")
    parser.add_argument("-o", "--output", help="File .txt output (mặc định: cùng tên input)")
    parser.add_argument("-w", "--workers", type=int, help="Số worker (mặc định: CPU - 1)")
    parser.add_argument("-t", "--tesseract", help="Đường dẫn tesseract.exe")
    parser.add_argument("-p", "--poppler", help="Đường dẫn Poppler (cho PDF)")
    args = parser.parse_args()

    ocr = FinancialOCR(
        tesseract_cmd=args.tesseract or TESSERACT_CMD,
        poppler_path=args.poppler or POPPLER_PATH,
    )
    text = ocr.process_input(args.input, args.output, args.workers)
    print(text)


if __name__ == "__main__":
    main()
