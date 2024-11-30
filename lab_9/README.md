# Lab 8

This project demonstrates how to detect and draw contours using OpenCV.

## Environment

Use your OpenCV virtual environment.

## Explanation and Observation

### 1. Load and Preprocess the Image

- **Function:** 
  ```python
  image = cv2.imread('input_image.jpg')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
- **Explanation:** The image is loaded in color, converted to grayscale, and then thresholded to obtain a binary image.
- **Observation:** Thresholding simplifies the image, making it easier to detect contours.

### 2. Find Contours

- **Function:**
    ```python
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

- **Explanation:** Finds contours in the binary image. cv2.RETR_EXTERNAL retrieves only the external contours, and cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
- **Observation:** Contours are detected and stored for further processing.

### 3. Draw Contours on the Original Image

- **Function:** `cv2.drawContours(image, contours, -1, (0, 255, 0), 2)`
- **Explanation:** Draws all the detected contours on the original image in green color with a thickness of 2 pixels.
- **Observation:** The original image now displays the contours, highlighting the detected shapes.

# _______________Additional example_______________

### 4. Calculate and Draw Convex Hulls

- **Function:** 
    ```for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(image, [hull], -1, (255, 0, 0), 2)

- **Explanation:** Calculates the convex hull for each contour, which is the smallest convex shape that completely encloses the contour. Draws the convex hulls on the image in blue color.
- **Observation:** Convex hulls are drawn around the detected objects, which can help in shape analysis and detecting convexity defects.

