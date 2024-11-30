# Lab 11

This project demonstrates adding Gaussian noise to an image and denoising it using different techniques in OpenCV.

## Environment

Use your OpenCV virtual environment.

## Explanation and Observation

### 1. Adding Gaussian Noise in Steps

- **Function:**
  ```python
  for i in range(5):  # 5 noise addition steps
      noise = np.random.normal(0, 0.1 * (i + 1), image.shape)  # Increase noise level
      noisy_image = np.clip(image + noise, 0, 1)  # Ensure values are within [0, 1]
- **Explanation:** Adds Gaussian noise to the image, increasing the noise level in each step. The np.random.normal function generates random values from a normal distribution, and the noise level is scaled by 0.1 * (i + 1).
- **Observation:** The image becomes progressively noisier with each step, simulating different levels of noise.

### 2. Denoising using Gaussian Blur

- **Function:**
  ```denoised_image_gaussian = cv2.GaussianBlur(noisy_image, (5, 5), 0)

- **Explanation:** Applies Gaussian Blur to the noisy image to reduce noise. The cv2.GaussianBlur function smoothens the image by averaging pixel values with a Gaussian kernel.
- **Observation:** The noise is reduced, but the image may become slightly blurred due to the smoothing effect.

# _______________Additional example_______________

### 2. Denoising using Non-Local Means Denoising

- **Function:** 
  ```denoised_image_nlm = cv2.fastNlMeansDenoising(
    (noisy_image * 255).astype('uint8'), None, h=10, templateWindowSize=7, searchWindowSize=21)

- **Explanation:** Applies Non-Local Means Denoising to the noisy image. This method removes noise while preserving edges by averaging pixels with similar intensity in a larger search window.
- **Observation:** The noise is effectively reduced while preserving the edges and details better than Gaussian Blur.