import cv2
import numpy as np
import pytesseract
import csv

img = cv2.imread("license_plate_2.jpg")
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

text = pytesseract.image_to_string(img)
print(text)