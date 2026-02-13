"""
Financial OCR - Trích xuất văn bản từ Báo cáo tài chính.
Hỗ trợ PDF (nhiều trang) và ảnh (PNG, JPG).
Tối ưu tốc độ với ProcessPoolExecutor (CPU-bound) + PaddleOCR cho tiếng Việt.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from core.exceptions import (
    DocumentNotFoundError,
    DocumentTypeNotSupportedError,
    OCREngineError,
)
from upload.image_loader import ImageLoader
from upload.pdf_loader import PdfLoader
from utils.worker import process_single_page


def _load_images(file_path: str | Path, config: Config) -> list["Image.Image"]:  # type: ignore[name-defined]
    """
    Load PDF or image files into a list of PIL.Image objects using
    the dedicated loader implementations.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PdfLoader(config=config)
    else:
        loader = ImageLoader()

    return loader.load(path)


def process_input(
    file_path: str | Path,
    config: Config,
    output_path: Optional[str | Path] = None,
    max_workers: Optional[int] = None,
) -> str:
    """
    Main OCR pipeline:
        Load -> Preprocess + OCR (song song, ProcessPoolExecutor) -> Ghi file.

    Returns:
        Toàn bộ text đã trích xuất từ tài liệu.
    """
    start_time = time.perf_counter()

    # Load images (PDF nhiều trang hoặc 1 ảnh).
    try:
        images = _load_images(file_path, config)
    except (DocumentNotFoundError, DocumentTypeNotSupportedError, ImportError) as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return ""

    if not images:
        print("[LỖI] Không có trang nào để xử lý.", file=sys.stderr)
        return ""

    # Chuẩn bị args cho worker (numpy array để picklable).
    worker_args = [
        (
            np.array(img),
            config.tesseract_cmd,
            config.ocr_lang,
            config.ocr_psm,
            config.ocr_oem,
            config.use_otsu,
        )
        for img in images
    ]

    # ProcessPoolExecutor: CPU-bound nên dùng Process.
    n_workers = max_workers or min(len(images), (os.cpu_count() or 4) - 1 or 1)
    texts: list[str] = [""] * len(images)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_single_page,
                    *args,
                ): i
                for i, args in enumerate(worker_args)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    texts[idx] = future.result()
                except (OCREngineError, Exception) as exc:  # noqa: BLE001
                    print(f"[LỖI] Trang {idx + 1}: {exc}", file=sys.stderr)
    except OCREngineError as exc:
        print(f"[LỖI] OCR Engine: {exc}", file=sys.stderr)
        return ""

    full_text = "\n\n".join(texts)
    elapsed = time.perf_counter() - start_time

    # Ghi file .txt (tên trùng input nếu không chỉ định khác).
    out_path = output_path or Path(file_path).with_suffix(".txt")
    Path(out_path).write_text(full_text, encoding="utf-8")

    print(
        f"Processed {len(images)} pages in {elapsed:.2f} seconds",
        file=sys.stderr,
    )
    return full_text


def main() -> None:
    """CLI đơn giản cho Financial OCR (PaddleOCR)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Financial OCR - Trích xuất văn bản từ Báo cáo tài chính",
    )
    parser.add_argument("input", help="File PDF hoặc ảnh (PNG, JPG)")
    parser.add_argument(
        "-o",
        "--output",
        help="File .txt output (mặc định: cùng tên input)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Số worker (mặc định: CPU - 1)",
    )
    parser.add_argument(
        "-p",
        "--poppler",
        help="Đường dẫn Poppler (cho PDF, dùng bởi pdf2image)",
    )
    args = parser.parse_args()

    # Khởi tạo Config từ CLI + default.
    config = Config()
    if args.poppler:
        config.poppler_path = args.poppler

    text = process_input(
        file_path=args.input,
        config=config,
        output_path=args.output,
        max_workers=args.workers,
    )
    print(text)


if __name__ == "__main__":
    main()
