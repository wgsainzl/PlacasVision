import cv2
import matplotlib.pyplot as plt
import numpy as np

#pip install easyocr

import easyocr
import pytesseract

placa1 = cv2.imread('placa_2.jpg')
placa1 = cv2.cvtColor(placa1, cv2.COLOR_BGR2RGB)

plt.imshow(placa1)
plt.axis('off')
plt.title('Imagen Original')
plt.show()

img_gauss = cv2.GaussianBlur(placa1, (15,15), 0)

plt.imshow(img_gauss)
plt.axis('off')
plt.title('Gaussian Blur')
plt.show()

av_filter = np.array([[1,1,1], [1,1,1], [1,1,1]]) / 9
img_av = cv2.filter2D(src=img_gauss, ddepth=-1, kernel=av_filter)
img_av_cv = cv2.blur(src=img_gauss, ksize=(5,5))

plt.imshow(img_av_cv)
plt.axis('off')
plt.title('Average')
plt.show()

img_hsv = cv2.cvtColor(img_av_cv, cv2.COLOR_RGB2HSV)
lower_hsv = np.array([0, 0, 0])
upper_hsv = np.array([180, 255, 65])
mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
mask_inv = cv2.bitwise_not(mask_hsv)
white_image = np.ones_like(img_gauss) * 255
result = cv2.bitwise_and(white_image, white_image, mask=mask_inv)

plt.imshow(result)
plt.axis('off')
plt.title('HSV')
plt.show()

try:
    reader = easyocr.Reader(['es', 'en'])
    results = reader.readtext(result)
    text_detected = "\n".join([text for _, text, _ in results])
    print("Texto detectado:")
    print(text_detected)

except Exception as e:
    print(f"Error al realizar OCR: {e}")