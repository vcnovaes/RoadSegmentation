import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Aplicar filtros

# Load the image in grayscale
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# Perform 2-level wavelet decomposition using the Haar wavelet
coeffs = pywt.dwt2(img, 'haar', mode='periodization')
LL, (LH, HL, HH) = coeffs
coeffs2 = pywt.dwt2(LL, 'haar', mode='periodization')
LL2, (LH2, HL2, HH2) = coeffs2

# Extract features from the wavelet coefficients
edges = LH + HL + HH
textures = LL2 + LH2 + HL2 + HH2

# Aplicar Bayes

# Threshold the features to obtain a binary mask
edges_thresh = cv2.threshold(edges, 500, 255, cv2.THRESH_BINARY)[1]
textures_thresh = cv2.threshold(textures, 590, 255, cv2.THRESH_BINARY)[1]

# Resize one of the binary masks to match the shape of the other
if edges_thresh.shape != textures_thresh.shape:
    edges_thresh = cv2.resize(edges_thresh, textures_thresh.shape[::-1])

# Combine the binary masks to obtain the final segmentation
segmentation = np.logical_or(edges_thresh, textures_thresh).astype(np.uint8)

# Display the original image and the segmented image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(segmentation, cmap='gray')
ax[1].set_title('Segmented Image')
plt.show()