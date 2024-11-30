import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('flower.png', cv2.IMREAD_GRAYSCALE)

# Apply simple thresholding
_, thresh_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive mean thresholding
thresh_adaptive_mean = cv2.adaptiveThreshold(image, 255, 
                                             cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)

# Apply adaptive Gaussian thresholding
thresh_adaptive_gaussian = cv2.adaptiveThreshold(image, 255, 
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 11, 2)

# Apply Otsu’s binarization
_, thresh_otsu = cv2.threshold(image, 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show results
plt.subplot(2, 2, 1), plt.imshow(thresh_simple, cmap='gray')
plt.title('Simple Thresholding')
plt.axis('off')

plt.subplot(2, 2, 2), plt.imshow(thresh_adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.axis('off')

plt.subplot(2, 2, 3), plt.imshow(thresh_adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.axis('off')

plt.subplot(2, 2, 4), plt.imshow(thresh_otsu, cmap='gray')
plt.title('Otsu’s Binarization')
plt.axis('off')

plt.tight_layout()
plt.show()

# _______________Additional example_______________

# Apply Gaussian blur before Otsu's binarization
blur = cv2.GaussianBlur(image, (5, 5), 0)
_, thresh_otsu_blur = cv2.threshold(blur, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the result
plt.imshow(thresh_otsu_blur, cmap='gray')
plt.title('Otsu’s Binarization after Gaussian Blur')
plt.axis('off')
plt.show()
