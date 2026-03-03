import json
import re
from typing import Any, Dict


def parse_financial_number(value: Any) -> float | None:
    """
    Chuẩn hóa số liệu tài chính (ưu tiên định dạng Việt Nam) về float.

    Hỗ trợ:
    - Dấu phân tách hàng nghìn kiểu Việt Nam: 1.234.567
    - Dấu thập phân kiểu Việt Nam: 1.234,56
    - Dấu thập phân kiểu quốc tế: 1,234.56 / 1234.56
    - Số âm dạng ngoặc kế toán: (1.234,56)
    - Chuỗi có khoảng trắng hoặc ký hiệu tiền tệ.

    - Nếu không chuyển được thì trả về None.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None

        # Xử lý dạng số âm theo chuẩn kế toán: (123) -> -123
        is_negative = txt.startswith("(") and txt.endswith(")")
        if is_negative:
            txt = txt[1:-1]

        # Loại bỏ khoảng trắng và ký tự không liên quan (giữ lại chữ số, dấu phẩy/chấm, dấu âm).
        txt = txt.replace(" ", "").replace("\u00A0", "")
        txt = re.sub(r"[^\d,.\-]", "", txt)
        if not txt:
            return None

        # Nếu có cả dấu "." và "," thì xác định ký tự thập phân theo vị trí xuất hiện cuối cùng.
        if "." in txt and "," in txt:
            if txt.rfind(",") > txt.rfind("."):
                # Việt Nam: 1.234,56
                txt = txt.replace(".", "").replace(",", ".")
            else:
                # Quốc tế: 1,234.56
                txt = txt.replace(",", "")
        elif "," in txt:
            # Chỉ có dấu phẩy: có thể là thập phân hoặc phân tách nghìn.
            integer_part, decimal_part = txt.rsplit(",", 1)
            if decimal_part.isdigit() and 0 < len(decimal_part) <= 2:
                txt = integer_part.replace(",", "") + "." + decimal_part
            else:
                txt = txt.replace(",", "")
        elif "." in txt:
            # Chỉ có dấu chấm: có thể là thập phân hoặc phân tách nghìn.
            integer_part, decimal_part = txt.rsplit(".", 1)
            if decimal_part.isdigit() and 0 < len(decimal_part) <= 2:
                txt = integer_part.replace(".", "") + "." + decimal_part
            else:
                txt = txt.replace(".", "")

        if is_negative and not txt.startswith("-"):
            txt = "-" + txt

        try:
            return float(txt)
        except ValueError:
            return None

    return None


def validate_financial_data(json_data: str, *, tolerance: float = 1e-2) -> str:
    """
    Kiểm định dữ liệu Báo cáo tài chính dựa trên các quy tắc kế toán cơ bản.

    Hiện tại triển khai rule:
    - total_assets == short_term_assets + long_term_assets (trong sai số cho phép).

    Nếu vi phạm, hàm sẽ:
    - Thêm key 'requires_human_review': true vào JSON.

    :param json_data: Chuỗi JSON đầu vào (từ LLM).
    :param tolerance: Sai số cho phép khi so sánh số học.
    :return: Chuỗi JSON sau khi được kiểm định/bổ sung thông tin.
    """
    try:
        data: Dict[str, Any] = json.loads(json_data) if json_data else {}
    except json.JSONDecodeError as exc:
        # Nếu JSON sai cấu trúc, đánh dấu cần review.
        return json.dumps({"raw": json_data, "requires_human_review": True}, ensure_ascii=False, indent=2)

    if not isinstance(data, dict):
        # Đảm bảo dữ liệu ở dạng dict trên cùng.
        data = {"value": data}

    total_assets = parse_financial_number(data.get("total_assets"))
    short_term_assets = parse_financial_number(data.get("short_term_assets"))
    long_term_assets = parse_financial_number(data.get("long_term_assets"))

    requires_review = False

    if (
        total_assets is not None
        and short_term_assets is not None
        and long_term_assets is not None
    ):
        if abs(total_assets - (short_term_assets + long_term_assets)) > tolerance:
            requires_review = True

    if requires_review:
        data["requires_human_review"] = True

    return json.dumps(data, ensure_ascii=False, indent=2)

