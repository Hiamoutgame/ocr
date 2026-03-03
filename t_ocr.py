#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import trio
from PIL import Image

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from module import LayoutRecognizer, TableStructureRecognizer, init_in_out
from module.layout_ordering import build_reading_order
from module.layout_to_markdown import render_page_markdown
from module.seeit import draw_box
from services.llm_mapper import map_to_json_schema

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' #2 gpus, uncontinuous
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' #1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # cpu


log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "t_ocr.log")

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


def create_ocr_engine(backend: str):
    # codex: Support choosing OCR recognizer backend without editing imports by hand.
    if backend == "onnx":
        try:
            from module.ocr_onnx import OCR as OCRONNX
            return OCRONNX()
        except Exception as exc:
            # codex: Graceful fallback when ONNX recognizer dependencies are not fully wired.
            print(f"[WARN] ONNX OCR backend unavailable, fallback to vietocr. Reason: {exc}")

    from module.ocr import OCR as OCRViet
    return OCRViet()


def run_ocr(ocr, img_np: np.ndarray, device_id: int = 0, return_time: bool = False):
    # codex: Normalize OCR call between VietOCR backend and ONNX backend signatures.
    if return_time:
        try:
            return ocr(img_np, device_id, return_time=True)
        except TypeError:
            result = ocr(img_np, device_id)
            return result, {"det": 0.0, "rec": 0.0, "cls": 0.0, "all": 0.0}

    return ocr(img_np, device_id)


def normalize_ocr_result(raw_result):
    # codex: ONNX backend can return fallback tuple on invalid input; keep one list format.
    if raw_result is None:
        return []
    if isinstance(raw_result, tuple):
        return []
    return raw_result


def normalize_layout_region(region: Dict) -> Dict:
    # codex: Convert layout output into one common schema consumed by markdown renderer.
    bbox = region.get(
        "bbox",
        [
            region.get("x0", 0),
            region.get("top", 0),
            region.get("x1", 0),
            region.get("bottom", 0),
        ],
    )
    x0, y0, x1, y1 = [int(float(v)) for v in bbox]
    return {
        "type": (region.get("type", "") or "").lower(),
        "score": float(region.get("score", 1.0) or 1.0),
        "bbox": [x0, y0, x1, y1],
        "content": "",
    }


def page_doc_key(output_path: str) -> str:
    # codex: Group page outputs by source document (xxx.pdf_3.jpg -> xxx.pdf).
    name = os.path.basename(output_path)
    return re.sub(r"_(\d+)\.jpg$", "", name)


def ocr_region_text(img: Image.Image, bbox: List[int], ocr, device_id: int) -> str:
    # codex: OCR region-level text for layout-preserving markdown generation.
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return ""

    region_img = img.crop((x0, y0, x1, y1))
    ocr_results = normalize_ocr_result(run_ocr(ocr, np.array(region_img), device_id=device_id))
    if not ocr_results:
        return ""

    text_lines = [t[0].strip() for _, t in ocr_results if t and t[0] and t[0].strip()]
    return "\n".join(text_lines).strip()


def extract_table_markdown(img: Image.Image, table_region: Dict, ocr, tsr: TableStructureRecognizer, device_id: int) -> str:
    # codex: Build markdown table from TSR structure + OCR cell text.
    if "bbox" in table_region:
        x0, y0, x1, y1 = map(int, table_region["bbox"])
    else:
        x0, y0, x1, y1 = map(
            int,
            [
                table_region.get("x0", 0),
                table_region.get("top", 0),
                table_region.get("x1", 0),
                table_region.get("bottom", 0),
            ],
        )

    if x1 <= x0 or y1 <= y0:
        return ""

    table_img = img.crop((x0, y0, x1, y1))
    tb_cpns = tsr([table_img])[0]
    boxes = normalize_ocr_result(run_ocr(ocr, np.array(table_img), device_id=device_id))
    if not boxes:
        return ""

    normalized_boxes = [
        {
            "x0": b[0][0],
            "x1": b[1][0],
            "top": b[0][1],
            "text": t[0],
            "bottom": b[-1][1],
            "layout_type": "table",
            "page_number": 0,
        }
        for b, t in boxes
        if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
    ]
    if not normalized_boxes:
        return ""

    avg_h = np.mean([b["bottom"] - b["top"] for b in normalized_boxes]) / 3
    boxes = LayoutRecognizer.sort_Y_firstly(normalized_boxes, avg_h)

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = LayoutRecognizer.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy
        )
        eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
        return LayoutRecognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted(
        [r for r in tb_cpns if re.match(r"table column$", r["label"])],
        key=lambda x: x["x0"],
    )
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

    return TableStructureRecognizer.construct_table(boxes, markdown=True)


def safe_markdown_to_json(markdown: str, model: str, base_url: str) -> str:
    # codex: Keep pipeline resilient if Ollama is down or model name is missing.
    try:
        return map_to_json_schema(markdown, model=model, base_url=base_url)
    except Exception as exc:
        return json.dumps(
            {
                "requires_human_review": True,
                "error": str(exc),
            },
            ensure_ascii=False,
            indent=2,
        )


def main(args):
    import torch.cuda

    cuda_devices = torch.cuda.device_count()
    limiter = [trio.CapacityLimiter(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    ocr = create_ocr_engine(args.ocr_backend)
    layout_recognizer = LayoutRecognizer("layout") if args.md_mode == "layout" else None
    table_recognizer = TableStructureRecognizer() if args.md_mode == "layout" else None

    images, outputs = init_in_out(args)
    doc_pages_md: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    def __ocr(i, device_id, img):
        print(f"Task {i} start")
        start_time = time.time()

        raw_bxs, time_dict = run_ocr(ocr, np.array(img), device_id=device_id, return_time=True)
        raw_bxs = normalize_ocr_result(raw_bxs)
        print(
            f"AI det={time_dict['det']:.2f}s | rec={time_dict['rec']:.2f}s | all={time_dict['all']:.2f}s"
        )

        bxs = [(line[0], line[1][0]) for line in raw_bxs]
        bxs = [
            {
                "text": t,
                "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
                "type": "ocr",
                "score": 1,
            }
            for b, t in bxs
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
        ]

        out_img = draw_box(images[i], bxs, ["ocr"], 1.0)
        out_img.save(outputs[i], quality=95)

        text_lines = [o["text"] for o in bxs]
        with open(outputs[i] + ".txt", "w+", encoding="utf-8") as f:
            f.write("\n".join(text_lines))

        if args.md_mode == "layout":
            # codex: Layout-aware markdown mode for scanned PDFs.
            layouts = layout_recognizer.forward([img], thr=float(args.threshold))[0]
            normalized_regions: List[Dict] = [normalize_layout_region(r) for r in layouts]

            for r in normalized_regions:
                rtype = r["type"]
                if rtype in {"header", "footer", "reference"}:
                    continue
                if rtype == "table":
                    r["content"] = extract_table_markdown(
                        img,
                        r,
                        ocr,
                        table_recognizer,
                        device_id=device_id,
                    )
                elif rtype == "figure":
                    r["content"] = ""
                else:
                    r["content"] = ocr_region_text(img, r["bbox"], ocr, device_id=device_id)

            # codex: Merge caption content into closest table/figure block.
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

            ordered_regions = build_reading_order(normalized_regions, img.size[0], img.size[1])
            markdown = render_page_markdown(ordered_regions)
            with open(outputs[i] + ".md", "w+", encoding="utf-8") as f:
                f.write(markdown)

            if args.debug_json:
                with open(outputs[i] + "_layout.json", "w+", encoding="utf-8") as f:
                    json.dump(ordered_regions, f, ensure_ascii=False, indent=2)

            key = page_doc_key(outputs[i])
            page_idx_match = re.search(r"_(\d+)\.jpg$", os.path.basename(outputs[i]))
            page_num = int(page_idx_match.group(1)) if page_idx_match else i
            doc_pages_md[key].append((page_num, markdown))

            if args.to_json:
                page_json = safe_markdown_to_json(
                    markdown,
                    model=args.llm_model,
                    base_url=args.ollama_base_url,
                )
                with open(outputs[i] + ".json", "w+", encoding="utf-8") as f:
                    f.write(page_json)
        else:
            # codex: Legacy OCR markdown mode (one bullet per OCR line).
            md_lines = [f"- {line}" for line in text_lines]
            with open(outputs[i] + ".md", "w+", encoding="utf-8") as f:
                f.write("\n".join(md_lines))

        elapsed = time.time() - start_time
        print(f"Task {i} done in {elapsed:.2f} seconds")

    async def __ocr_thread(i, device_id, img, limiter_obj=None):
        if limiter_obj:
            async with limiter_obj:
                print(f"Task {i} use device {device_id}")
                await trio.to_thread.run_sync(lambda: __ocr(i, device_id, img))
        else:
            __ocr(i, device_id, img)

    async def __ocr_launcher():
        # codex: Force sequential mode for layout/json path to avoid race conditions on shared accumulators.
        if args.md_mode == "layout" or args.to_json:
            for i, img in enumerate(images):
                await __ocr_thread(i, 0, img)
            return

        if cuda_devices > 1:
            async with trio.open_nursery() as nursery:
                for i, img in enumerate(images):
                    nursery.start_soon(
                        __ocr_thread,
                        i,
                        i % cuda_devices,
                        img,
                        limiter[i % cuda_devices],
                    )
                    await trio.sleep(0.1)
        else:
            for i, img in enumerate(images):
                await __ocr_thread(i, 0, img)

    trio.run(__ocr_launcher)

    if args.md_mode == "layout":
        # codex: Write document-level merged markdown/json for downstream structured extraction.
        for key, pages in doc_pages_md.items():
            pages.sort(key=lambda x: x[0])
            combined_md = "\n\n".join(md for _, md in pages if md and md.strip()).strip()
            if not combined_md:
                continue

            out_md = os.path.join(args.output_dir, f"{key}_layout.md")
            with open(out_md, "w+", encoding="utf-8") as f:
                f.write(combined_md)
            logging.info(f"Saved combined markdown: {out_md}")

            if args.to_json:
                combined_json = safe_markdown_to_json(
                    combined_md,
                    model=args.llm_model,
                    base_url=args.ollama_base_url,
                )
                out_json = os.path.join(args.output_dir, f"{key}_layout.json")
                with open(out_json, "w+", encoding="utf-8") as f:
                    f.write(combined_json)
                logging.info(f"Saved combined json: {out_json}")

    print("OCR hoan thanh!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        help="Folder/file input image or PDF",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Output folder (default: ./ocr_outputs)",
        default="./ocr_outputs",
    )
    parser.add_argument(
        "--md_mode",
        choices=["layout", "ocr"],
        default="layout",
        help="layout: keep document layout in markdown; ocr: bullet list markdown.",
    )
    parser.add_argument(
        "--threshold",
        default=0.2,
        type=float,
        help="Layout confidence threshold (used in md_mode=layout).",
    )
    parser.add_argument(
        "--ocr_backend",
        choices=["vietocr", "onnx"],
        default="vietocr",
        help="OCR backend for text recognition.",
    )
    parser.add_argument(
        "--to_json",
        action="store_true",
        help="Convert markdown output to JSON using Ollama model.",
    )
    parser.add_argument(
        "--llm_model",
        default="llama3.1:latest",
        help="Ollama model name, e.g. llama3.1:latest.",
    )
    parser.add_argument(
        "--ollama_base_url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--debug_json",
        action="store_true",
        help="Write *_layout.json with ordered layout regions.",
    )

    main(parser.parse_args())
