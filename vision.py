import cv2
import matplotlib.pyplot as plt
import numpy as np

#pip install easyocr # Install library for OCR

import easyocr # Import library aftet installation

placa1 = cv2.imread('placa_2.jpg') # Read image file

# Convert image from BGR (OpenCV default) to RGB for visualization
placa1 = cv2.cvtColor(placa1, cv2.COLOR_BGR2RGB) 


# Display the original image
plt.imshow(placa1)
plt.axis('off')
plt.title('Imagen Original')
plt.show()

# Apply Gaussian Blur to reduce noise
img_gauss = cv2.GaussianBlur(placa1, (15,15), 0)

# Display Gaussian blurred image
plt.imshow(img_gauss)
plt.axis('off')
plt.title('Gaussian Blur')
plt.show()

# Define a 3x3 averaging kernel and apply it using filter2D
av_filter = np.array([[1,1,1], [1,1,1], [1,1,1]]) / 9
img_av = cv2.filter2D(src=img_gauss, ddepth=-1, kernel=av_filter)

# Apply OpenCV's built-in averaging filter (box blur)
img_av_cv = cv2.blur(src=img_gauss, ksize=(5,5))

# Display the averaged image
plt.imshow(img_av_cv)
plt.axis('off')
plt.title('Average')
plt.show()

# Convert the image to HSV color space
img_hsv = cv2.cvtColor(img_av_cv, cv2.COLOR_RGB2HSV)

# Define HSV threshold range to isolate dark areas
lower_hsv = np.array([0, 0, 0])
upper_hsv = np.array([180, 255, 65])
mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

# Invert the mask to highlight light areas
mask_inv = cv2.bitwise_not(mask_hsv)

# Create a white image of the same size
white_image = np.ones_like(img_gauss) * 255

# Apply the inverted mask to the white image
result = cv2.bitwise_and(white_image, white_image, mask=mask_inv)


# Display the result after HSV filtering
plt.imshow(result)
plt.axis('off')
plt.title('HSV')
plt.show()

# Try Optical Character Recognition using EasyOCR
try:
    # Initialize EasyOCR Reader with Spanish and English support
    reader = easyocr.Reader(['es', 'en'])

    # Run OCR on the result image
    results = reader.readtext(result)

    # Extract and print detected text
    text_detected = "\n".join([text for _, text, _ in results])
    print("Texto detectado:")
    print(text_detected)

except Exception as e:
    print(f"Error al realizar OCR: {e}")