import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt

filename = 'valid\994141_sat.jpg'
# Load the satellite image
img = cv2.imread(filename, 0)

# Perform the 2D Discrete Wavelet Transform on the image
img = cv2.GaussianBlur(img, (1, 1), 0)
coeffs = pywt.dwt2(img, 'haar')

# Extract the subbands from the coefficients
LL, (LH, HL, HH) = coeffs

# Apply a high-pass filter to the HH subband
HH_filt = cv2.GaussianBlur(HH, (3, 3), 0) - HH

# Convert the filtered subband to 8-bit unsigned integer format
HH_filt = cv2.normalize(HH_filt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Apply a threshold to the filtered HH subband to create a mask
# mask = cv2.threshold(HH_filt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Apply the mask to the original image to extract the road features
# road_features = cv2.bitwise_and(img, HH_filt)

# # Save the resulting road features image
# cv2.imwrite('road_features.jpg', road_features)

fig = plt.figure(figsize=(12, 3))
# imgs = []
ax = fig.add_subplot(1, 4, 1)
ax.imshow(LH + HL,
          interpolation="nearest", cmap=plt.cm.gray)
# ax.imshow(, interpolation="nearest", cmap=plt.cm.gray)
ax.set_title("HL", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
plt.show()
