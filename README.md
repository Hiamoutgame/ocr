<p align="center">
  <a href="./README.md">Tiếng Việt</a> |
  <a href="./README_en.md">English</a>
</p>

# DeepDoc + VietOCR — OCR tiếng Việt, nhận layout & bảng

Công cụ OCR chạy được trên CPU, kết hợp **detection ONNX** (Paddle-style), **VietOCR** (nhận chữ tiếng Việt), **Layout Recognizer** và **Table Structure Recognizer** (YOLOv10 ONNX) để xuất văn bản và Markdown giữ bố cục gần với tài liệu gốc.

---

## Mục lục

- [Cài đặt](#cài-đặt)
- [Cách chạy](#cách-chạy)
- [Các script và khi nào dùng](#các-script-và-khi-nào-dùng)
- [Kiến trúc kỹ thuật](#kiến-trúc-kỹ-thuật)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Cài đặt

**Yêu cầu:** Python 3.10+

```bash
git clone <repo_url>
cd ocr
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Mô hình ONNX (detection, layout, TSR) sẽ được tải từ Hugging Face khi chạy lần đầu (hoặc đặt sẵn trong thư mục `onnx/`).

---

## Cách chạy

| Script | Mục đích | Ví dụ |
|--------|----------|--------|
| **t_ocr.py** | Chỉ OCR (detect + nhận chữ) | `python t_ocr.py --inputs ./pdfs --output_dir ./ocr_outputs` |
| **t_recognizer.py** | Chỉ layout hoặc chỉ TSR (vẽ box, xuất bảng) | `python t_recognizer.py --inputs ./pdfs --mode layout --threshold 0.2 --output_dir ./layout_out` |
| **full_pipeline.py** | Layout + OCR + bảng → **1 file Markdown gần PDF** | `python full_pipeline.py --inputs ./pdfs --output_dir ./layout_md_outputs --threshold 0.2` |

**Đầu vào:** `--inputs` có thể là:
- Một file ảnh hoặc PDF
- Một thư mục chứa ảnh/PDF (duyệt đệ quy)

**Đầu ra:**
- **t_ocr:** Ảnh vẽ box + `.txt` + `.md` (bullet) theo từng trang/ảnh.
- **t_recognizer:** Ảnh vẽ vùng layout/TSR; với `--mode tsr` thêm file `.md` bảng (cả ảnh = 1 bảng).
- **full_pipeline:** Mỗi trang có `*_full.md`; mỗi tài liệu có **`<tên_tài_liệu>_layout.md`** (một file Markdown gộp toàn bộ trang, giữ thứ tự đọc 1/2 cột, title, đoạn, bảng, figure). Có thể thêm `--debug_json` để xuất `*_layout.json` (vùng layout đã sắp thứ tự).

---

## Các script và khi nào dùng

| Nhu cầu | Script |
|--------|--------|
| Chỉ lấy text nhanh, không cần cấu trúc | **t_ocr.py** |
| Xem vùng layout (title, text, table, figure…) hoặc trích 1 bảng từ ảnh | **t_recognizer.py** (layout hoặc tsr) |
| Xuất **1 file Markdown cả tài liệu**, bố cục gần PDF (title, đoạn, bảng, figure) | **full_pipeline.py** |

---

## Kiến trúc kỹ thuật

### OCR

- **Detection:** Mô hình PaddleOCR chuyển sang ONNX (detect vùng chữ).
- **Recognition:** VietOCR (mặc định: vgg_seq2seq, CPU). Có thể dùng bản ONNX qua `module/ocr_onnx.py` để tăng tốc (đổi import trong script).
- Pipeline: Ảnh → resize/normalize → detection ONNX → crop từng dòng → VietOCR → text.

Tham khảo kiến trúc Paddle PP-OCR (ví dụ [PP-OCRv5](https://arxiv.org/html/2507.05595v1)). Phần recognition gốc đã được thay bằng VietOCR để tối ưu cho tiếng Việt; chuyển VietOCR sang ONNX tham khảo [bài viết Viblo](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw).

### Layout Recognizer & Table Structure Recognizer

- **Layout:** YOLOv10 ONNX — nhận diện 10 loại: Text, Title, Figure, Figure caption, Table, Table caption, Header, Footer, Reference, Equation.
- **TSR:** Cùng backbone YOLOv10 — nhận diện cấu trúc bảng (column, row, header, spanning cell…) để dựng lại bảng dạng Markdown.

Chi tiết YOLOv10: [arxiv](https://arxiv.org/pdf/2405.14458).

### Full pipeline (layout-preserving Markdown)

- `module/layout_ordering.py`: Sắp thứ tự đọc (1 cột / 2 cột) cho các vùng layout.
- `module/layout_to_markdown.py`: Render từng loại vùng (title → `#`, text → đoạn, table → markdown bảng, figure → placeholder + caption) thành Markdown.
- `full_pipeline.py`: Gọi Layout Recognizer → OCR từng vùng → TSR cho vùng bảng → reading order → render → ghi `*_full.md` và `<doc>_layout.md`.

---

## Tài liệu tham khảo

- [DeepDoc (RAGFlow)](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md)
- [PP-OCRv5](https://arxiv.org/html/2507.05595v1)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [YOLOv10](https://arxiv.org/pdf/2405.14458)
