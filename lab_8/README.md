# Lab 7

This project demonstrates different edge detection techniques using OpenCV.

## Environment

Use your OpenCV virtual environment.

## Explanation and Observation

### 1. Sobel Edge Detection

- **Function:** `sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  #Vertical edges  sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)`
- **Explanation:** Applies the Sobel operator to detect horizontal and vertical edges and then combines them.
- **Observation:** Sobel edge detection is simple and fast, good for detecting both horizontal and vertical edges, but produces thicker edges and is sensitive to noise.

### 2. Canny Edge Detection

- **Function:** `cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)`
- **Explanation:** Calculates the threshold for small regions by taking the mean of neighborhood pixel values.
- **Observation:** Handles varying lighting conditions better than simple thresholding.

### 3. Adaptive Gaussian Thresholding

- **Function:** `edges = cv2.Canny(image, 100, 200)`
- **Explanation:** Applies Canny edge detection, which includes noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding.
- **Observation:** Canny edge detection gives thinner, well-defined edges but requires experimenting with two threshold values.

# _______________Additional example_______________

### 5. Otsu’s Binarization after Gaussian Blur

- **Function:** `blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  edges_blurred = cv2.Canny(blurred_image, 50, 150)`

- **Explanation:** Applies Gaussian blur to reduce noise before performing Canny edge detection.
- **Observation:** Reducing noise with Gaussian blur before edge detection can result in cleaner edges and reduce false detections caused by noise.

