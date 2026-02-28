## Prompt for Cursor AI — docTR Line-Level Layout Detection (CPU, grayscale/color, with fallback) + VietOCR → RAG Markdown

### Context

I need a fully local pipeline for scanned Vietnamese financial PDFs (multi-page).
Primary goals:

* Detect text at **LINE level** (not word-level output; you can merge word boxes into lines).
* Support both **grayscale and color scans** (auto-handle).
* Have a **fallback** if docTR detection fails or returns too few lines.
* Use **VietOCR** for recognition (Vietnamese optimized).
* Output **RAG-optimized Markdown** (semantic headings + bullet key:value), not pixel-perfect layout.
* Must run **CPU-first** (deploy friendly), no cloud.

### Strict pipeline order (MANDATORY)

PDF → page images
→ preprocess (auto grayscale/color normalize, no heavy cleaning)
→ **docTR detection (line-level)**
→ fallback detection if needed
→ crop lines
→ VietOCR recognition (batch, CPU)
→ semantic parsing (label/value/section)
→ render Markdown (RAG-first)

### DO NOT

* Do not use cloud OCR APIs
* Do not use Tesseract as primary
* Do not use LLM for parsing
* Do not output raw TXT dump
* Do not reconstruct pixel spacing / ASCII tables

---

## Deliverables (Create these Python modules)

Python 3.10+, type hints, docstrings, error handling.

### 1) `pdf_loader.py`

Convert PDF pages to images (prefer PyMuPDF).

```python
def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """Return ordered list of page image paths."""
```

### 2) `preprocess.py`

Support grayscale & color automatically:

* If color: convert to RGB (consistent)
* Create an additional grayscale version for detection if it improves results
* Light normalization only (no heavy denoise since I already have a base)
  Return both representations so detector can choose best.

```python
import numpy as np

def load_and_normalize(image_path: str) -> dict:
    """
    Return:
      {
        "rgb": np.ndarray(H,W,3) uint8,
        "gray": np.ndarray(H,W) uint8
      }
    """
```

### 3) `layout_detector_doctr.py` (CORE)

Use docTR **detection** model to produce bounding boxes.
Requirements:

* Must output **line-level** boxes (x0,y0,x1,y1) in reading order.
* docTR may return word boxes — merge them into lines.
* Run on CPU.
* Provide confidence filtering.

```python
from typing import List, Tuple

Box = Tuple[int,int,int,int]

def detect_lines_doctr(img_rgb: "np.ndarray", img_gray: "np.ndarray") -> List[Box]:
    """
    Use docTR detector.
    Strategy:
      1) Try detection on grayscale (common for scans)
      2) If result weak, try on rgb
    Return line-level boxes sorted top->bottom then left->right.
    """
```

**Line merging logic (MANDATORY)**
If docTR returns word boxes:

* Cluster boxes into lines using y-overlap / baseline proximity:

  * Two boxes are in same line if vertical IoU > threshold OR |y_center diff| < threshold
* Merge line box as min(x0,y0), max(x1,y1)
* Within each line, sort by x0 to keep reading order.

### 4) `layout_fallback.py` (MANDATORY fallback)

If docTR returns too few lines or too low coverage:

* Fallback detector options (choose 1 that is CPU-friendly and easy):

  * OpenCV morphological line detection + connected components
  * OR CRAFT (if acceptable dependency)

Provide a simple robust fallback:

```python
def detect_lines_fallback(img_gray: "np.ndarray") -> list[Box]:
    """Return line-level boxes using OpenCV heuristics as fallback."""
```

### 5) Fallback trigger policy (IMPORTANT)

In `layout_detector.py` orchestrator:

* Run docTR first.
* If any of these true, trigger fallback:

  * number of lines < MIN_LINES (e.g., 15)
  * total text area coverage < MIN_COVERAGE (e.g., 0.02 of page ar_
