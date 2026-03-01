<p align="center">
  <a href="./README.md">Tiếng Việt</a> |
  <a href="./README_en.md">English</a>
</p>

# DeepDoc + VietOCR — Vietnamese OCR with layout & table support

CPU-friendly OCR pipeline combining **ONNX text detection** (Paddle-style), **VietOCR** (Vietnamese recognition), **Layout Recognizer**, and **Table Structure Recognizer** (YOLOv10 ONNX) to output text and layout-preserving Markdown.

---

## Contents

- [Installation](#installation)
- [How to run](#how-to-run)
- [Scripts and when to use them](#scripts-and-when-to-use-them)
- [Technical architecture](#technical-architecture)
- [References](#references)

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone <repo_url>
cd ocr
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

ONNX models (detection, layout, TSR) are downloaded from Hugging Face on first run (or place them in `onnx/`).

---

## How to run

| Script | Purpose | Example |
|--------|--------|--------|
| **t_ocr.py** | OCR only (detect + recognize) | `python t_ocr.py --inputs ./pdfs --output_dir ./ocr_outputs` |
| **t_recognizer.py** | Layout or TSR only (draw boxes, export table) | `python t_recognizer.py --inputs ./pdfs --mode layout --threshold 0.2 --output_dir ./layout_out` |
| **full_pipeline.py** | Layout + OCR + tables → **one layout-preserving Markdown file** | `python full_pipeline.py --inputs ./pdfs --output_dir ./layout_md_outputs --threshold 0.2` |

**Input:** `--inputs` can be:
- A single image or PDF file
- A directory of images/PDFs (traversed recursively)

**Output:**
- **t_ocr:** Image with drawn boxes + `.txt` + `.md` (bullet list) per page/image.
- **t_recognizer:** Image with layout/TSR boxes; with `--mode tsr` adds a table `.md` (whole image as one table).
- **full_pipeline:** Per-page `*_full.md`; per document **`<document_name>_layout.md`** (single Markdown file for the whole document, reading order 1/2 column, title, paragraphs, tables, figures). Use `--debug_json` to also write `*_layout.json` (ordered layout regions).

---

## Scripts and when to use them

| Need | Script |
|------|--------|
| Quick text extraction, no structure | **t_ocr.py** |
| Inspect layout regions (title, text, table, figure…) or extract one table from an image | **t_recognizer.py** (layout or tsr) |
| **One Markdown file per document** with layout close to the PDF | **full_pipeline.py** |

---

## Technical architecture

### OCR

- **Detection:** PaddleOCR-style model in ONNX (text region detection).
- **Recognition:** VietOCR (default: vgg_seq2seq, CPU). Optional ONNX recognizer in `module/ocr_onnx.py` for faster inference (change import in scripts).
- Pipeline: Image → resize/normalize → ONNX detection → crop lines → VietOCR → text.

See Paddle PP-OCR architecture (e.g. [PP-OCRv5](https://arxiv.org/html/2507.05595v1)). Original recognition is replaced by VietOCR for Vietnamese; VietOCR-to-ONNX conversion: [Viblo article](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw).

### Layout Recognizer & Table Structure Recognizer

- **Layout:** YOLOv10 ONNX — 10 classes: Text, Title, Figure, Figure caption, Table, Table caption, Header, Footer, Reference, Equation.
- **TSR:** Same YOLOv10 backbone — table structure (column, row, header, spanning cell…) for Markdown table output.

Details: [YOLOv10 arxiv](https://arxiv.org/pdf/2405.14458).

### Full pipeline (layout-preserving Markdown)

- `module/layout_ordering.py`: Reading-order (1-column / 2-column) for layout regions.
- `module/layout_to_markdown.py`: Renders region types (title → `#`, text → paragraph, table → markdown table, figure → placeholder + caption) to Markdown.
- `full_pipeline.py`: Runs Layout Recognizer → OCR per region → TSR for table regions → reading order → render → writes `*_full.md` and `<doc>_layout.md`.

---

## References

- [DeepDoc (RAGFlow)](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md)
- [PP-OCRv5](https://arxiv.org/html/2507.05595v1)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [YOLOv10](https://arxiv.org/pdf/2405.14458)
