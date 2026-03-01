from __future__ import annotations

from typing import Dict, List


def _is_two_column(text_regions: List[Dict], page_width: int) -> bool:
    """Heuristic two-column detection based on x-centers spread."""
    if len(text_regions) < 6 or page_width <= 0:
        return False
    centers = sorted((r["bbox"][0] + r["bbox"][2]) / 2.0 for r in text_regions)
    mid = len(centers) // 2
    left_med = centers[mid // 2]
    right_med = centers[mid + (len(centers) - mid) // 2]
    return (right_med - left_med) > (0.22 * page_width)


def _sort_top_left(regions: List[Dict]) -> List[Dict]:
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def build_reading_order(regions: List[Dict], page_width: int, page_height: int) -> List[Dict]:
    """
    Build page reading order with simple 1-col / 2-col handling.

    In two-column mode: left column top->bottom then right column top->bottom.
    """
    if not regions:
        return []

    text_like_types = {"text", "title", "table", "figure", "equation"}
    text_like = [r for r in regions if r.get("type", "") in text_like_types]
    if not text_like:
        return _sort_top_left(regions)

    if not _is_two_column(text_like, page_width):
        return _sort_top_left(regions)

    pivot = page_width / 2.0
    left_col = []
    right_col = []
    for r in regions:
        x0, _, x1, _ = r["bbox"]
        cx = (x0 + x1) / 2.0
        if cx <= pivot:
            left_col.append(r)
        else:
            right_col.append(r)

    ordered = _sort_top_left(left_col) + _sort_top_left(right_col)
    return ordered

