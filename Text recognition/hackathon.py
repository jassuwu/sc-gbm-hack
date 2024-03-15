import cv2
import pytesseract
from pytesseract import Output
from ultralytics import YOLO

img = cv2.imread("sample4.jpg")
pytesseract.pytesseract.tesseract_cmd = (
    "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# h, w, c = img.shape
thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
binary_image = cv2.adaptiveThreshold(
    thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)
cv2.imshow("vd", binary_image)
inv_binary_image = cv2.bitwise_not(binary_image)
cv2.imshow("vh", inv_binary_image)
d = pytesseract.image_to_data(inv_binary_image, output_type=Output.DICT)
extracted_text = pytesseract.image_to_string(inv_binary_image)

print(extracted_text.strip())
cv2.imshow("img", img)

_3d_img = cv2.cvtColor(inv_binary_image, cv2.COLOR_GRAY2RGB)
d = pytesseract.image_to_data(_3d_img, output_type=Output.DICT)
n_boxes = len(d["text"])
for i in range(n_boxes):
    if int(d["conf"][i]) > -7:
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        img = cv2.rectangle(_3d_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = d["text"][i]
        cv2.putText(
            img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
print(text)
cv2.imshow("imgg", _3d_img)
cv2.imshow("final", img)
cv2.waitKey(0)
