import cv2
import numpy as np


def remove_red_blue_stamps(image_array: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh để xóa con dấu/chữ ký màu đỏ và xanh dương.

    Ý tưởng:
    - Chuyển ảnh sang không gian màu HSV để tách màu tốt hơn so với RGB/BGR.
    - Tạo mask cho các dải màu đỏ và xanh dương thường dùng trong con dấu/chữ ký.
    - Với các pixel thuộc mask, thay thế bằng màu trắng để không làm nhiễu OCR.

    :param image_array: Ảnh đầu vào dạng numpy array (kỳ vọng RGB từ PIL).
    :return: Ảnh đã được làm sạch dấu, dạng numpy array RGB.
    """
    if image_array is None or image_array.size == 0:
        return image_array

    # Pipeline chính đọc ảnh qua PIL nên dữ liệu đầu vào là RGB.
    # Chuyển trực tiếp RGB -> HSV để lọc màu bằng OpenCV.
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Dải màu đỏ trong HSV có hai vùng (quanh 0 và quanh 180 độ hue)
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Dải màu xanh dương (blue) – thường thấy ở chữ ký
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Kết hợp mask đỏ và xanh
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Tùy chọn: mở/đóng morphology để làm sạch nhiễu nhỏ
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Thay các pixel của con dấu/chữ ký bằng nền trắng.
    clean_img = image_array.copy()
    clean_img[mask > 0] = [255, 255, 255]

    return clean_img

