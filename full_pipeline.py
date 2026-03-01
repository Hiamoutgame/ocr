import logging
import os
import sys
import argparse
import numpy as np
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import json

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from module.ocr import OCR 
from module import LayoutRecognizer, TableStructureRecognizer, init_in_out
from module.layout_ordering import build_reading_order
from module.layout_to_markdown import render_page_markdown

from datetime import datetime

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "full_pipeline.log")

# Count previous runs by counting lines that start with "=== Run"
run_count = 1
if os.path.exists(log_file):    
    with open(log_file, "r", encoding="utf-8") as f:
        run_count += sum(1 for line in f if line.startswith("=== Run"))

# Write run header with count and date
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n=== Run {run_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

sys.stdout = open(log_file, "a", encoding="utf-8")
sys.stderr = sys.stdout

def extract_table_markdown(img, table_region, ocr):
    # Use bbox if present
    if "bbox" in table_region:
        x0, y0, x1, y1 = map(int, table_region["bbox"])
    else:
        x0, y0, x1, y1 = map(int, [table_region["x0"], table_region["top"], table_region["x1"], table_region["bottom"]])
    table_img = img.crop((x0, y0, x1, y1))
    tb_cpns = TableStructureRecognizer()([table_img])[0]
    boxes = ocr(np.array(table_img))
    boxes = LayoutRecognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "layout_type": "table",
          "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3
    )

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = LayoutRecognizer.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
        eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
        return LayoutRecognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted([r for r in tb_cpns if re.match(
        r"table column$", r["label"])], key=lambda x: x["x0"])
    clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    markdown = TableStructureRecognizer.construct_table(boxes, markdown=True)
    return markdown


def normalize_layout_region(region: Dict) -> Dict:
    """Normalize layout region into common dict schema."""
    bbox = region.get("bbox", [region.get("x0", 0), region.get("top", 0), region.get("x1", 0), region.get("bottom", 0)])
    x0, y0, x1, y1 = [int(float(v)) for v in bbox]
    return {
        "type": (region.get("type", "") or "").lower(),
        "score": float(region.get("score", 1.0) or 1.0),
        "bbox": [x0, y0, x1, y1],
        "content": "",
    }


def ocr_region_text(img, bbox: List[int], ocr: OCR) -> str:
    """OCR text content from one region bbox."""
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return ""
    region_img = img.crop((x0, y0, x1, y1))
    ocr_results = ocr(np.array(region_img))
    if not ocr_results:
        return ""
    text_lines = [t[0].strip() for _, t in ocr_results if t and t[0] and t[0].strip()]
    return "\n".join(text_lines).strip()


def page_doc_key(output_path: str) -> str:
    """
    Group per-page outputs by source document key.
    Example: xxx.pdf_0.jpg -> xxx.pdf
    """
    name = os.path.basename(output_path)
    return re.sub(r"_(\d+)\.jpg$", "", name)

def main(args):
    images, outputs = init_in_out(args)
    print(f"Loaded {len(images)} images")
    print(f"Output paths: {outputs}")
    layout_recognizer = LayoutRecognizer("layout")
    ocr = OCR()
    doc_pages_md: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for idx, img in enumerate(images):
        print(f"Processing image {idx}: {outputs[idx]}")
        start_time = time.time()  # <-- Start timing

        layouts = layout_recognizer.forward([img], thr=float(args.threshold))[0]
        print(f"Detected {len(layouts)} layout regions")
        normalized_regions: List[Dict] = [normalize_layout_region(r) for r in layouts]

        # Fill region content by type
        for r in normalized_regions:
            rtype = r["type"]
            if rtype in {"header", "footer", "reference"}:
                continue
            if rtype == "table":
                print(f"Extracting table markdown for region: {r['bbox']}")
                r["content"] = extract_table_markdown(img, r, ocr)
            elif rtype == "figure":
                # figure body is placeholder; caption might be a nearby separate region
                r["content"] = ""
            else:
                r["content"] = ocr_region_text(img, r["bbox"], ocr)

        # Attach nearby figure/table captions to owner block
        for caption in [x for x in normalized_regions if x["type"] in {"figure caption", "table caption"}]:
            cx0, cy0, cx1, cy1 = caption["bbox"]
            best = None
            best_dist = 10**9
            for owner in normalized_regions:
                if caption["type"] == "figure caption" and owner["type"] != "figure":
                    continue
                if caption["type"] == "table caption" and owner["type"] != "table":
                    continue
                ox0, oy0, ox1, oy1 = owner["bbox"]
                x_overlap = max(0, min(cx1, ox1) - max(cx0, ox0))
                y_dist = min(abs(cy0 - oy1), abs(oy0 - cy1))
                score = y_dist - (0.1 * x_overlap)
                if score < best_dist:
                    best_dist = score
                    best = owner
            if best and caption.get("content"):
                if best.get("content"):
                    best["content"] = f"{best['content']}\n{caption['content']}"
                else:
                    best["content"] = caption["content"]

        # Build reading order and render page markdown
        ordered_regions = build_reading_order(normalized_regions, img.size[0], img.size[1])
        markdown_concat = render_page_markdown(ordered_regions)

        # Page-level markdown output
        out_path = outputs[idx] + "_full.md"
        print(f"Writing concatenated markdown to: {out_path}")
        with open(out_path, "w+", encoding='utf-8') as f:
            f.write(markdown_concat)
        logging.info(f"Saved concatenated markdown to: {out_path}")

        # Optional debug regions
        if args.debug_json:
            with open(outputs[idx] + "_layout.json", "w+", encoding="utf-8") as f:
                json.dump(ordered_regions, f, ensure_ascii=False, indent=2)

        # Accumulate for document-level markdown
        key = page_doc_key(outputs[idx])
        page_idx_match = re.search(r"_(\d+)\.jpg$", os.path.basename(outputs[idx]))
        page_num = int(page_idx_match.group(1)) if page_idx_match else idx
        doc_pages_md[key].append((page_num, markdown_concat))

        elapsed = time.time() - start_time  # <-- End timing
        print(f"Processing image {idx} done in {elapsed:.2f} seconds")  # <-- Print elapsed time

    # Write one combined markdown per document
    for key, pages in doc_pages_md.items():
        pages.sort(key=lambda x: x[0])
        combined = "\n\n---\n\n".join(p[1] for p in pages if p[1].strip())
        doc_out = os.path.join(args.output_dir, f"{key}_layout.md")
        with open(doc_out, "w+", encoding="utf-8") as f:
            f.write(combined.strip())
        print(f"Wrote document markdown: {doc_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory or file path for images or PDFs",
                        required=True)
    parser.add_argument('--output_dir', help="Directory for output markdown files. Default: './table_markdown_outputs'",
                        default="./table_markdown_outputs")
    parser.add_argument('--threshold',
                        help="Detection threshold. Default: 0.5",
                        default=0.5)
    parser.add_argument('--debug_json', action='store_true',
                        help="Write ordered layout regions to per-page JSON for debugging.")
    args = parser.parse_args()
    main(args)