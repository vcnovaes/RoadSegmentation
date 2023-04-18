import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# Load the image and extract features


def get_label(x, y, filename):
    # Read the corresponding mask image
    mask = cv2.imread(f'target/{filename}.png', cv2.IMREAD_GRAYSCALE)

    # Determine if the pixel is part of a road or not
    if mask[y, x] > 0:
        return 1  # Road pixel
    else:
        return 0  # Non-road pixel


filename = "104"
image = cv2.imread(f"input/{filename}.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
features = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
features = np.int0(features)

# Create training data
train_data = []
train_labels = []
for i in range(len(features)):
    x, y = features[i][0]
    # Custom function to get the label for a given pixel
    label = get_label(x, y, filename)
    # Extract a small patch around the pixel
    patch = gray[y-10:y+10, x-10:x+10]
    # Add the flattened patch as a feature vector
    train_data.append(patch.flatten())
    train_labels.append(label)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
mask = cv2.imread(f'target/{filename}.png', cv2.IMREAD_GRAYSCALE)
lab = []
print(mask.shape)
for i in range(0, mask.shape[0]):
    for j in range(0, mask.shape[1]):
        lab.append(get_label(i, j, filename))
print(mask[1, 1])
knn.fit(train_data[0].reshape(1, -1), train_labels)

# Segment the image using KNN
segmented = np.zeros_like(gray)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        # Extract a small patch around the pixel
        patch = gray[y-10:y+10, x-10:x+10]
        feature_vector = patch.flatten().reshape(1, -1)
        label = knn.predict(feature_vector)
        segmented[y, x] = label

# Display the segmented image
# cv2.imshow('Segmented', segmented)
plt.imshow(segmented)
cv2.waitKey(0)
plt.show()
