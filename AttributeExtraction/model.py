import os
import pandas as pd
import numpy as np
import cv2
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
# import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from keras import layers, models, optimizers


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def preprocess_images(images):
    images = [cv2.resize(img, (32, 32)) for img in images]
    images = np.array(images) / 255.0  
    return images


def load_labels_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    labels = df['IDENTITY'].values
    return labels


def preprocess_labels(labels):

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return labels_encoded


image_folder = 'train_v2/train'
images = load_images_from_folder(image_folder)


images = preprocess_images(images)


csv_file = 'written_name_train_v2.csv'
labels = load_labels_from_csv(csv_file)


labels = preprocess_labels(labels)
num_classes = len(np.unique(labels))


X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


test_loss, test_accuracy = model.evaluate(X_val, y_val)
print('Test Accuracy:', test_accuracy)


model.save('handwritten_recognition_model.h5')
