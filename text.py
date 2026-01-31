import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def preprocess(img):
    """Tiền xử lý ảnh để tăng độ chính xác OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Giảm nhiễu, giữ rõ viền chữ
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    return denoised


def img_to_text(img_path):
    """Chuyển ảnh sang text với cấu hình tối ưu."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    img = preprocess(img)

    # PSM 7 = single text line (tối ưu cho 1 dòng chữ)
    config = r"--psm 7 --oem 3"
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()

text = img_to_text(".//assets//image.png")
print(text)