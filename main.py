import argparse
import os
from typing import List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from services.image_cleaner import remove_red_blue_stamps
from services.table_extractor import extract_financial_tables
from services.llm_mapper import map_to_json_schema
from services.validator import validate_financial_data


def load_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """
    Đọc file PDF và chuyển từng trang thành ảnh PIL.

    Sử dụng PyMuPDF để render từng trang thành raster image.
    """
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def load_images_from_input_path(input_path: str) -> List[Image.Image]:
    """
    Chuẩn hóa đầu vào thành danh sách ảnh PIL.

    - Nếu input là PDF -> render từng trang.
    - Nếu input là một file ảnh đơn (png/jpg/jpeg/tif/...) -> đọc thành một ảnh.
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        return load_images_from_pdf(input_path)

    # Trường hợp là file ảnh
    img = Image.open(input_path).convert("RGB")
    return [img]


def preprocess_images(images: List[Image.Image]) -> List[Image.Image]:
    """
    Áp dụng bước tiền xử lý để xóa dấu đỏ/xanh trên từng ảnh.

    - Chuyển ảnh PIL -> numpy array (RGB).
    - Gọi remove_red_blue_stamps.
    - Chuyển kết quả về PIL.Image cho các bước OCR tiếp theo.
    """
    cleaned: List[Image.Image] = []
    for img in images:
        arr = np.array(img)
        arr_clean = remove_red_blue_stamps(arr)
        img_clean = Image.fromarray(arr_clean)
        cleaned.append(img_clean)
    return cleaned


def run_pipeline(input_path: str, output_path: str) -> None:
    """
    Chạy toàn bộ pipeline:
    1. Đọc file PDF/ảnh thành danh sách ảnh.
    2. Tiền xử lý xóa dấu đỏ/xanh.
    3. Trích xuất tất cả bảng tài chính thành Markdown.
    4. Gọi LLM (Ollama) để map sang JSON schema.
    5. Kiểm định lại các quan hệ kế toán cơ bản.
    6. Ghi kết quả cuối cùng ra file JSON.
    """
    print(f"[1/5] Đọc file đầu vào: {input_path}")
    images = load_images_from_input_path(input_path)
    if not images:
        raise RuntimeError("Không đọc được trang nào từ file đầu vào.")
    print(f"   -> Số trang ảnh: {len(images)}")

    print("[2/5] Tiền xử lý ảnh (xóa dấu đỏ/xanh)...")
    cleaned_images = preprocess_images(images)

    print("[3/5] Trích xuất bảng tài chính từ layout...")
    markdown_tables = extract_financial_tables(cleaned_images)
    if not markdown_tables:
        print("   -> Không tìm thấy bảng tài chính nào. Kết quả sẽ là JSON rỗng.")

    print("[4/5] Gọi LLM (Ollama) để map sang JSON schema...")
    json_from_llm = map_to_json_schema(markdown_tables)

    print("[5/5] Kiểm định dữ liệu tài chính (cross-validation)...")
    validated_json = validate_financial_data(json_from_llm)

    print(f"Ghi kết quả cuối cùng ra: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(validated_json)

    print("Hoàn tất pipeline phân tích Báo cáo tài chính.")


def main() -> None:
    """
    Điểm vào chính cho CLI.

    Ví dụ chạy:
        python main.py --input baocao.pdf
    """
    parser = argparse.ArgumentParser(
        description="Pipeline phân tích Báo cáo tài chính từ PDF/ảnh -> JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn tới file PDF hoặc ảnh đầu vào.",
    )
    parser.add_argument(
        "--output",
        default="result.json",
        help="Đường dẫn file JSON kết quả (mặc định: result.json).",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    run_pipeline(input_path, output_path)


if __name__ == "__main__":
    main()

