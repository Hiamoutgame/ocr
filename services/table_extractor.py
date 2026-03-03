import re
from typing import List

import numpy as np
from PIL import Image

from module.ocr import OCR
from module import LayoutRecognizer, TableStructureRecognizer


def extract_table_markdown(img: Image.Image, table_region: dict, ocr: OCR) -> str:
    """
    HÃ m con trÃ­ch xuáº¥t má»™t báº£ng Ä‘Æ¡n láº» thÃ nh Markdown.

    Logic Ä‘Æ°á»£c tÃ¡i sá»­ dá»¥ng/tÆ°Æ¡ng tá»± tá»« `full_pipeline.py`:
    - Cáº¯t áº£nh theo bounding box cá»§a báº£ng.
    - DÃ¹ng TableStructureRecognizer Ä‘á»ƒ láº¥y cáº¥u trÃºc báº£ng.
    - DÃ¹ng OCR Ä‘á»ƒ nháº­n diá»‡n text trong tá»«ng Ã´.
    - DÃ¹ng TableStructureRecognizer.construct_table Ä‘á»ƒ build Markdown.
    """
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

    table_img = img.crop((x0, y0, x1, y1))
    tb_cpns = TableStructureRecognizer()([table_img])[0]
    boxes = ocr(np.array(table_img))
    boxes = LayoutRecognizer.sort_Y_firstly(
        [
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
        ],
        np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3,
    )

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

    markdown = TableStructureRecognizer.construct_table(boxes, markdown=True)
    return markdown


def extract_financial_tables(images: List[Image.Image]) -> str:
    """
    TrÃ­ch xuáº¥t táº¥t cáº£ cÃ¡c báº£ng tÃ i chÃ­nh tá»« danh sÃ¡ch áº£nh vÃ  ná»‘i láº¡i thÃ nh Markdown.

    Quy trÃ¬nh:
    - Khá»Ÿi táº¡o LayoutRecognizer("layout") vÃ  OCR().
    - Vá»›i tá»«ng áº£nh:
      + Cháº¡y layout_recognizer.forward Ä‘á»ƒ láº¥y cÃ¡c vÃ¹ng layout.
      + Lá»c cÃ¡c vÃ¹ng cÃ³ type == "table".
      + Vá»›i má»—i vÃ¹ng báº£ng, gá»i extract_table_markdown Ä‘á»ƒ láº¥y Markdown.
    - GhÃ©p toÃ n bá»™ Markdown báº£ng láº¡i thÃ nh má»™t chuá»—i vÄƒn báº£n.

    :param images: Danh sÃ¡ch áº£nh (PIL.Image) Ä‘Ã£ Ä‘Æ°á»£c lÃ m sáº¡ch dáº¥u.
    :return: Chuá»—i Markdown chá»©a táº¥t cáº£ cÃ¡c báº£ng tÃ i chÃ­nh.
    """
    if not images:
        return ""

    layout_recognizer = LayoutRecognizer("layout")
    ocr = OCR()

    all_tables_md: List[str] = []

    for page_idx, img in enumerate(images):
        layouts = layout_recognizer.forward([img])[0]

        table_regions = [
            r
            for r in layouts
            if str(r.get("type", "")).lower() == "table"
            or str(r.get("layout_type", "")).lower() == "table"
        ]

        for tbl_idx, region in enumerate(table_regions):
            try:
                md = extract_table_markdown(img, region, ocr)
                if md and md.strip():
                    header = f"### Báº£ng {tbl_idx + 1} - Trang {page_idx + 1}\n\n"
                    all_tables_md.append(header + md.strip())
            except Exception:
                # Náº¿u cÃ³ lá»—i á»Ÿ má»™t báº£ng Ä‘Æ¡n láº», ta bá» qua Ä‘á»ƒ khÃ´ng lÃ m há»ng cáº£ pipeline.
                continue

    return "\n\n\n".join(all_tables_md).strip()



