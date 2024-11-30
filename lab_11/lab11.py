import cv2
import numpy as np
import torch
import torch.nn as nn

# Load original image
image = cv2.imread('flower.png', cv2.IMREAD_GRAYSCALE)

# Create noise
noise = np.random.normal(0, 25, image.shape).astype('uint8')

# Add noise to create adversarial image
adversarial_image = cv2.addWeighted(image, 1.0, noise, 0.1, 0)

cv2.imshow('Adversarial Image', adversarial_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define latent dimension
latent_dim = 100

# Generator model
generator = nn.Sequential(
    nn.Linear(latent_dim, 128),  # Maps input noise to 128 features
    nn.ReLU(),                   # Activation function introduces non-linearity
    nn.Linear(128, 256),         # Expands feature space to 256 dimensions
    nn.ReLU(),                   # Non-linear activation
    nn.Linear(256, 28 * 28),     # Output layer for 28x28 image
    nn.Tanh()                    # Scales output to range [-1, 1]
)

# Discriminator model
discriminator = nn.Sequential(
    nn.Linear(28 * 28, 256),     # Input layer for 28x28 image
    nn.LeakyReLU(0.2),           # Activation with slope 0.2
    nn.Linear(256, 128),         # Hidden layer reduces features to 128
    nn.LeakyReLU(0.2),           # Non-linear activation
    nn.Linear(128, 1),           # Output layer for real/fake score
    nn.Sigmoid()                 # Outputs probability [0, 1]
)

# Assume the following variables are defined elsewhere:
# real_imgs, fake_imgs, real_labels, fake_labels, criterion, optimizer_D, optimizer_G

# Calculate loss for real images
real_loss = criterion(discriminator(real_imgs), real_labels)  # Discriminator classifies real images

# Calculate loss for fake images
fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)  # Discriminator classifies fake images

# Total loss for the Discriminator
d_loss = real_loss + fake_loss  # Combine real and fake losses

# Backpropagation for the Discriminator
d_loss.backward()
optimizer_D.step()  # Update Discriminator weights

# Generator updates
fake_preds = discriminator(fake_imgs)  # Discriminator's predictions on fake images
g_loss = criterion(fake_preds, real_labels)  # Generator tries to fool the Discriminator

# Backpropagation for the Generator
g_loss.backward()
optimizer_G.step()  # Update Generator weights

# Load and preprocess the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128)) / 255.0  # Normalize to range [0, 1]

# Add Gaussian noise in steps
for i in range(5):  # 5 noise addition steps
    noise = np.random.normal(0, 0.1 * (i + 1), image.shape)  # Increase noise level
    noisy_image = np.clip(image + noise, 0, 1)  # Ensure values are within [0, 1]
    cv2.imshow(f"Step {i+1}", (noisy_image * 255).astype('uint8'))  # Show noisy image
    cv2.waitKey(0)  # Wait for user input

# Denoising using GaussianBlur
denoised_image_gaussian = cv2.GaussianBlur(noisy_image, (5, 5), 0)
cv2.imshow("Denoised Image - Gaussian Blur", (denoised_image_gaussian * 255).astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# _______________Additional example_______________

# Denoising using Non-Local Means Denoising
denoised_image_nlm = cv2.fastNlMeansDenoising(
    (noisy_image * 255).astype('uint8'), None, h=10, templateWindowSize=7, searchWindowSize=21)
cv2.imshow("Denoised Image - Non-Local Means", denoised_image_nlm)
cv2.waitKey(0)
cv2.destroyAllWindows()
