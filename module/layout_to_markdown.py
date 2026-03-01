from __future__ import annotations

from typing import Dict, List


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def merge_lines_to_paragraph(text: str) -> str:
    lines = [normalize_whitespace(ln) for ln in (text or "").splitlines() if normalize_whitespace(ln)]
    return " ".join(lines).strip()


def render_region(region: Dict) -> str:
    """Render one region dict to markdown chunk."""
    rtype = (region.get("type") or "").lower()
    content = (region.get("content") or "").strip()
    if rtype in {"header", "footer", "reference"}:
        return ""
    if rtype == "title":
        text = merge_lines_to_paragraph(content)
        return f"# {text}" if text else ""
    if rtype == "table":
        return content
    if rtype == "figure":
        cap = merge_lines_to_paragraph(content)
        if cap:
            return f"![Figure](#)\n\n*{cap}*"
        return "![Figure](#)"
    if rtype in {"figure caption", "table caption"}:
        text = merge_lines_to_paragraph(content)
        return f"*{text}*" if text else ""
    if rtype == "equation":
        text = merge_lines_to_paragraph(content)
        if not text:
            return ""
        return f"```text\n{text}\n```"
    if rtype == "text":
        return merge_lines_to_paragraph(content)
    return merge_lines_to_paragraph(content)


def render_page_markdown(page_regions: List[Dict]) -> str:
    """Render ordered regions to one page markdown."""
    chunks: List[str] = []
    for region in page_regions:
        chunk = render_region(region)
        if chunk:
            chunks.append(chunk)
    return "\n\n".join(chunks).strip()

