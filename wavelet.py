from skimage.filters import sobel
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# # Aplicar filtros
img_file = 'valid/994141_sat.jpg'

# # Load the image in grayscale
# img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

# # Perform 2-level wavelet decomposition using the Haar wavelet
# coeffs = pywt.dwt2(img, 'haar', mode='periodization')
# LL, (LH, HL, HH) = coeffs
# coeffs2 = pywt.dwt2(LL, 'haar', mode='periodization')
# LL2, (LH2, HL2, HH2) = coeffs2

# # Extract features from the wavelet coefficients
# edges = LH + HL + HH
# textures = LL2 + LH2 + HL2 + HH2

# # Aplicar Bayes

# # Threshold the features to obtain a binary mask
# edges_thresh = cv2.threshold(edges, 500, 255, cv2.THRESH_BINARY)[1]
# textures_thresh = cv2.threshold(textures, 590, 255, cv2.THRESH_BINARY)[1]

# # Resize one of the binary masks to match the shape of the other
# if edges_thresh.shape != textures_thresh.shape:
#     edges_thresh = cv2.resize(edges_thresh, textures_thresh.shape[::-1])

# # Combine the binary masks to obtain the final segmentation
# segmentation = np.logical_or(edges_thresh, textures_thresh).astype(np.uint8)

# # Display the original image and the segmented image
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original Image')
# ax[1].imshow(segmentation, cmap='gray')
# ax[1].set_title('Segmented Image')
# plt.show()
# Load image
original = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


soaps_image = plt.imread(img_file)

# Make the image grayscale
soaps_image_gray = color.rgb2gray(soaps_image)

# apply edge detection filters
edge_sobel = sobel(soaps_image_gray)
edge_sobel = 255 - edge_sobel
# coeffs2 = pywt.dwt2(edge_sobel, 'db1', 'zero', (1, 0))
# LL, (LH, HL, HH) = coeffs2
# Show original image
show_image(edge_sobel, 'Original')


# # Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# coeffs2 = pywt.dwt2(original, 'db1', 'zero', (1, 0))
# LL, (LH, HL, HH) = coeffs2

# fig = plt.figure(figsize=(12, 3))
# imgs = []
# # imgs.append()
# ax = fig.add_subplot(1, 4, 1)
# ax.imshow(LH + HL,
#           interpolation="nearest", cmap=plt.cm.gray)
# # ax.imshow(, interpolation="nearest", cmap=plt.cm.gray)
# ax.set_title("HL", fontsize=10)
# ax.set_xticks([])
# ax.set_yticks([])

# fig.tight_layout()
# plt.show()


# # Carregar a imagem
# image = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

# # Aplicar o ajuste de histograma

# # Exibir a imagem original e a imagem com contraste ajustado
# cv2.imshow('Imagem Original', image)
# cv2.imshow('Imagem com Contraste Ajustado', equ)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
