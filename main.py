import cv2
import pytesseract 

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread(".//assets//Screenshot 2026-01-27 152032.png")