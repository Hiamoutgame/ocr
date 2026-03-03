import json
import os
from typing import Any, Dict

import requests


def _build_prompt(markdown_tables_text: str) -> str:
    """
    Xây dựng prompt tiếng Việt cho LLM phân tích Báo cáo tài chính.

    Prompt được thiết kế để:
    - Giải thích rõ vai trò: chuyên gia kế toán.
    - Yêu cầu mapping các chỉ tiêu tiếng Việt sang JSON keys chuẩn hóa.
    - Yêu cầu chỉ trả về JSON hợp lệ, không có giải thích thêm.
    """
    prompt = (
        "Bạn là một chuyên gia kế toán. Dưới đây là dữ liệu Báo cáo tài chính "
        "dạng Markdown được trích xuất từ PDF. Hãy phân tích và trích xuất các chỉ tiêu ra JSON.\n"
        "Yêu cầu: Map các chỉ tiêu tiếng Việt (VD: Tiền, Tài sản ngắn hạn, Phải trả người bán...) "
        "thành JSON keys chuẩn hóa (VD: cash, short_term_assets, accounts_payable). "
        "Đầu ra CHỈ LÀ JSON hợp lệ, không giải thích.\n"
        f"Dữ liệu Markdown:\n{markdown_tables_text}"
    )
    return prompt


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    """
    Gọi API Ollama đang chạy local để sinh JSON.

    :param prompt: Prompt đầy đủ gửi tới LLM.
    :param model: Tên model Ollama (ví dụ: 'llama3', 'qwen2.5', ...).
    :param base_url: URL gốc của Ollama, vd 'http://localhost:11434'.
    :return: Chuỗi phản hồi từ LLM (mong đợi là JSON).
    """
    url = base_url.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    data = resp.json()

    # Theo spec của Ollama, nội dung chính nằm trong field "response"
    text = data.get("response", "")
    return text


def map_to_json_schema(markdown_tables_text: str, *, model: str | None = None, base_url: str | None = None) -> str:
    """
    Dùng LLM (Ollama) để map dữ liệu bảng (Markdown) sang JSON schema tài chính.

    - Gọi API Ollama local qua HTTP.
    - Prompt yêu cầu LLM chỉ trả về JSON hợp lệ.
    - Hàm trả về string JSON (đã được strip), để bước sau xử lý tiếp.

    Có thể cấu hình:
    - Model qua biến môi trường OLLAMA_MODEL (mặc định: 'huihui_ai/hunyuan-mt-abliterated:latest' – tên phải trùng với `ollama list`).
    - URL qua biến môi trường OLLAMA_BASE_URL (mặc định: 'http://localhost:11434').
    """
    if not markdown_tables_text or not markdown_tables_text.strip():
        # Không có dữ liệu để map
        return json.dumps({})

    # Tên model phải trùng tuyệt đối với kết quả `ollama list`, ví dụ:
    # 'gpt-oss:latest' hoặc 'huihui_ai/hunyuan-mt-abliterated:latest'
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
    # Lấy URL từ tham số truyền vào, nếu không có thì dùng biến môi trường OLLAMA_BASE_URL,
    # cuối cùng fallback về địa chỉ mặc định của Ollama local.
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    prompt = _build_prompt(markdown_tables_text)
    raw_text = call_ollama(prompt, model=model, base_url=base_url)

    # Đảm bảo chắc chắn trả về JSON string hợp lệ:
    # - Thử parse -> dict -> dump lại để loại bỏ prefix/suffix thừa nếu có.
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Nếu model trả về kèm text thừa, cố gắng tìm đoạn JSON lớn nhất.
        # Đây là xử lý "cứu hộ nhẹ" để pipeline không hỏng hoàn toàn.
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM output is not valid JSON: {exc}\nRaw: {raw_text}") from exc
        else:
            raise ValueError(f"LLM output is not valid JSON.\nRaw: {raw_text}")

    return json.dumps(parsed, ensure_ascii=False, indent=2)

