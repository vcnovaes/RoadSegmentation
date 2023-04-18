import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

# Define paths
train_dir = ""
input_dir = os.path.join(train_dir, "input")
target_dir = os.path.join(train_dir, "target")
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define image size
img_height = 256
img_width = 256

# Load the data
input_imgs = []
target_imgs = []
for filename in os.listdir(input_dir):
    try:
        input_img = load_img(os.path.join(input_dir, filename),
                             target_size=(img_height, img_width))
        target_img = load_img(os.path.join(target_dir, filename), target_size=(
            img_height, img_width), color_mode="grayscale")
        input_imgs.append(img_to_array(input_img))
        target_imgs.append(img_to_array(target_img))
    except:
        print("Erro")
input_imgs = np.array(input_imgs) / 255.0
target_imgs = np.array(target_imgs) / 255.0

# Define the model
inputs = Input(shape=(img_height, img_width, 3))
conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D((2, 2))(drop4)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
drop5 = Dropout(0.5)(conv5)
up6 = UpSampling2D((2, 2))(drop5)
up6 = Conv2D(256, (2, 2), activation="relu", padding="same")(up6)
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(merge6)
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
up7 = UpSampling2D((2, 2))(conv6)
up7 = Conv2D(128, (2, 2), activation="relu", padding="same")(up7)
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(merge7)
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
up8 = UpSampling2D((2, 2))(conv7)
up8 = Conv2D(64, (2, 2), activation="relu", padding="same")(up8)
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(merge8)
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
up9 = UpSampling2D((2, 2))(conv8)
up9 = Conv2D(32, (2, 2), activation="relu", padding="same")(up9)
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(merge9)
conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
conv9 = Conv2D(2, (3, 3), activation="relu", padding="same")(conv9)
outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(lr=1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(input_imgs, target_imgs, batch_size=32,
                    epochs=50, validation_split=0.1)


model.save(os.path.join(model_dir, "road_segmentation_model.h5"))
