# Financial OCR - Setup & Run Guide

## Project Overview

**Financial OCR** is a Python application for extracting text from financial reports. It supports PDF (multi-page) and image files (PNG, JPG). The project uses parallel processing (`ProcessPoolExecutor`) for CPU-bound OCR tasks.

**Current Version:** `1.0.0`  
**Last Updated:** February 2025

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ | Required for type hints (`list[X]`) |
| Tesseract OCR | 5.x | Must be installed separately |
| Poppler | - | Required for PDF support (Windows) |

---

## System Dependencies

### 1. Tesseract OCR

Download and install Tesseract:

- **Windows:** [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS:** `brew install tesseract tesseract-lang`
- **Linux:** `sudo apt install tesseract-ocr tesseract-ocr-vie` (Ubuntu/Debian)

**Vietnamese + English:** Install language packs (`vie`, `eng`) if needed.

### 2. Poppler (for PDF support on Windows)

- **Windows:** Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- Extract to e.g. `C:\Program Files\poppler-25.12.0\`
- Add `Library\bin` to PATH or set `POPPLER_PATH` in config

- **macOS:** `brew install poppler`
- **Linux:** `sudo apt install poppler-utils`

---

## Setup

### 1. Clone or navigate to the project

```bash
cd ocr
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Python dependencies

```bash
pip install opencv-python numpy pytesseract Pillow pdf2image
```

Or use `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure paths (if needed)

Edit `text.py` and adjust these variables at the top:

```python
# Windows - Tesseract path (set to None if in PATH)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Windows - Poppler path for PDF (set to None if in PATH)
POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"
```

---

## Running the Project

### CLI

```bash
# Process a single image
python text.py assets/image.png

# Process a PDF
python text.py assets/CvResume.pdf

# Specify output file
python text.py assets/image.png -o output.txt

# Custom number of workers (parallel pages)
python text.py report.pdf -w 4

# Override Tesseract path
python text.py image.png -t "C:\Program Files\Tesseract-OCR\tesseract.exe"

# Override Poppler path (for PDF)
python text.py document.pdf -p "C:\Program Files\poppler\Library\bin"
```

### As a Python module

```python
from text import FinancialOCR

ocr = FinancialOCR()
text = ocr.process_input("assets/CvResume.pdf")
# Output saved to assets/CvResume.txt
# Prints: "Processed N pages in X.XX seconds"
```

### Custom configuration

```python
from text import FinancialOCR

ocr = FinancialOCR(
    tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    poppler_path=r"C:\Program Files\poppler\Library\bin",
    lang="vie+eng",
    psm=6,           # 6 = uniform block, 3 = fully automatic
    pdf_dpi=300,
    use_otsu=True,   # Otsu threshold (False = Adaptive)
)
text = ocr.process_input("report.pdf", output_path="result.txt")
```

---

## Project Structure

```
ocr/
├── text.py          # Main OCR module (FinancialOCR class)
├── assets/          # Sample images and PDFs
├── SETUP.md         # This file
├── README.md
└── requirements.txt # Python dependencies (optional)
```

---

## Supported Formats

| Input | Extension | Notes |
|-------|-----------|-------|
| PDF | `.pdf` | Requires `pdf2image` + Poppler |
| Image | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif` | Via PIL |

---

## Output

- Extracted text is written to a `.txt` file with the same base name as the input (e.g. `report.pdf` → `report.txt`)
- Use `-o` / `--output` to specify a custom output path
- Processing time is printed to stderr: `Processed N pages in X.XX seconds`

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `TesseractNotFoundError` | Install Tesseract and set `TESSERACT_CMD` or add to PATH |
| `ImportError: pdf2image` | Run `pip install pdf2image` |
| PDF conversion fails | Install Poppler and set `POPPLER_PATH` on Windows |
| Poor OCR quality | Try `use_otsu=False` (Adaptive threshold) or increase `pdf_dpi` |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `opencv-python` (cv2) | Image preprocessing (grayscale, threshold) |
| `pytesseract` | OCR engine |
| `Pillow` (PIL) | Image I/O |
| `pdf2image` | PDF → image conversion |
| `numpy` | Array operations |
| `concurrent.futures` | Parallel processing (ProcessPoolExecutor) |
