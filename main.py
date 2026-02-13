"""
Financial OCR - Main Entry Point.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# --- QUAN TRỌNG: FIX LỖI CRASH KHI MULTIPROCESSING ---
# Phải đặt trước khi import bất kỳ thư viện deep learning nào (numpy, paddle, torch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Giới hạn mỗi process Paddle chỉ dùng 1 luồng CPU để tránh tranh chấp tài nguyên
# Vì chúng ta đã dùng ProcessPoolExecutor để chia việc rồi.
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from PIL import Image

from config import Config
from core.exceptions import (
    DocumentNotFoundError,
    DocumentTypeNotSupportedError,
    OCREngineError,
)
from upload.image_loader import ImageLoader
from upload.pdf_loader import PdfLoader
from utils.worker import process_single_page

def _load_images(file_path: str | Path, config: Config) -> list[Image.Image]:
    """
    Load PDF or image files into a list of PIL.Image objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    # Factory logic đơn giản
    if ext == ".pdf":
        loader = PdfLoader(config=config)
    elif ext in ImageLoader.SUPPORTED_EXTENSIONS:
        loader = ImageLoader()
    else:
        raise DocumentTypeNotSupportedError(f"Unsupported file type: {ext}")

    return loader.load(path)


def process_input(
    file_path: str | Path,
    config: Config,
    output_path: Optional[str | Path] = None,
    max_workers: Optional[int] = None,
) -> str:
    start_time = time.perf_counter()

    # 1. Load Images
    print(f"Loading document: {file_path}...", file=sys.stderr)
    try:
        images = _load_images(file_path, config)
    except Exception as exc:
        print(f"[ERROR] Loading failed: {exc}", file=sys.stderr)
        return ""

    if not images:
        print("[ERROR] No pages loaded.", file=sys.stderr)
        return ""

    print(f"Loaded {len(images)} pages. Preparing workers...", file=sys.stderr)

    # 2. Prepare Data for Multiprocessing
    # Convert PIL Image -> Numpy Array để có thể Pickle (gửi qua Process khác)
    worker_args = [
        (
            np.array(img),
            config.ocr_lang,
        )
        for img in images
    ]

    # 3. Determine Workers Strategy
    # PaddleOCR tốn RAM, nên không dùng hết CPU core. 
    # Mặc định an toàn là 2 hoặc 4 workers.
    default_workers = 2 
    if os.cpu_count() and os.cpu_count() > 4:
        default_workers = 4
    
    n_workers = max_workers or min(len(images), default_workers)
    
    texts: list[str] = [""] * len(images)
    print(f"Starting OCR with {n_workers} worker processes...", file=sys.stderr)

    # 4. Execute Parallel Processing
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Map future to page index
            future_to_idx = {
                executor.submit(process_single_page, *args): i
                for i, args in enumerate(worker_args)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    texts[idx] = result
                    print(f"✓ Completed page {idx + 1}/{len(images)}", file=sys.stderr)
                except Exception as exc:
                    print(f"✗ Failed page {idx + 1}: {exc}", file=sys.stderr)
                    
    except KeyboardInterrupt:
        print("\n[STOP] Process interrupted by user.", file=sys.stderr)
        return ""

    # 5. Save Results
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(texts)
    
    if output_path:
        out_path = Path(output_path)
    else:
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"{Path(file_path).stem}.txt"

    out_path.write_text(full_text, encoding="utf-8")
    
    elapsed = time.perf_counter() - start_time
    print(f"\nAll done! Processed {len(images)} pages in {elapsed:.2f}s", file=sys.stderr)
    print(f"Output saved to: {out_path}", file=sys.stderr)
    
    return full_text


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Financial OCR (PaddleOCR)")
    parser.add_argument("input", help="Input file (PDF/Image)")
    parser.add_argument("-o", "--output", help="Output .txt file")
    parser.add_argument("-w", "--workers", type=int, help="Number of workers (default: auto)")
    parser.add_argument("-p", "--poppler", help="Poppler bin path")
    args = parser.parse_args()

    config = Config()
    if args.poppler:
        config.poppler_path = args.poppler

    process_input(args.input, config, args.output, args.workers)

if __name__ == "__main__":
    main()