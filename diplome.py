# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from tensorflow.keras.callbacks import ModelCheckpoint

"""# Load the Data"""

from google.colab import drive

drive.mount('/content/drive/')

# !unzip "/content/drive/MyDrive/Colab Notebooks/Нейронки/Alz_data.zip" -d "/content/drive/MyDrive/Colab Notebooks/Нейронки/alz/"

dir  = "/content/drive/MyDrive/Учеба/gamedev/НИР/train/"
font = "/content/drive/MyDrive/Учеба/gamedev/НИР/Arial.ttf"
dir_test = "/content/drive/MyDrive/Учеба/gamedev/НИР/test/"

logsModel = "/content/drive/MyDrive/Учеба/gamedev/НИР/logs/"

modelFilePath = '/content/drive/MyDrive/Учеба/gamedev/НИР/models/'

img_height = 208
img_width = 176

dataset = [f.name for f in os.scandir(dir) if f.is_dir()]
count = len(dataset)
print(count)
print(dataset)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size= 16
data_dir = dir

# Rescaling the Input Image
train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes = dataset,
    subset='training',
    shuffle=True) # set as training data

validation_generator = train_datagen.flow_from_directory(
    data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes = dataset,
    subset='validation',
    shuffle=True) # set as validation data

"""# Visualising samples from training set"""

training_samples, labels = train_generator.next()

plt.figure(figsize=(60, 60))

for i in range(10):
    plt.subplot(5, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    image = training_samples[i]
    plt.imshow(image, cmap=plt.cm.binary)
    label = int(np.argmax(labels[i]))
    plt.ylabel(dataset[label])
plt.show()

"""# The model"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import TensorBoard
import datetime

"""**Visualising the model**"""

!pip install visualkeras

"""## Optimized Model"""

model = Sequential()

model.add(Convolution2D(16, (2, 2), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(count, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

"""**Visualising the model**"""

import visualkeras
from PIL import ImageFont

font = ImageFont.truetype(font, 12, encoding="unic")
visualkeras.layered_view(model, legend=True, font=font)

"""### Training"""

filepath = modelFilePath
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1, save_best_only=True,mode='max')

log_dir = logsModel + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [checkpoint, tensorboard_callback]

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=10,
          verbose=1,
          callbacks=callbacks)

"""## Evaluate"""

loaded_model = tf.keras.models.load_model(filepath)

test_loss, test_accuracy = loaded_model.evaluate(validation_generator)
print("Validation Loss: {}, Validation Accuracy: {}".format(test_loss, test_accuracy))

"""# Test the model"""

from tensorflow.keras import layers

test_ds = tf.keras.utils.image_dataset_from_directory(dir,
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

def plot_image(i, predictions_array, true_label, img):
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = 'green' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(dataset[predicted_label],
                                100*np.max(predictions_array),
                                dataset[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    plt.xticks(range(count), dataset)
    plt.yticks([])
    thisplot = plt.bar(range(count), predictions_array, color="#7d4646")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

"""Модель 2"""

for images, labels in normalized_ds.take(1):
    for i in range(10):
        image = images[i]
        label = labels[i]
        image_batch = tf.expand_dims(image, [0])
        prediction = loaded_model.predict(image_batch)

        # Plotting
        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        plot_image(i, prediction[0], label, image)
        plt.subplot(1,2,2)
        plot_value_array(i, prediction[0],  label)
        plt.show()

"""Модель 1"""

model = Sequential()

model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.5))


model.add(Dense(count, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

import visualkeras
from PIL import ImageFont

font = ImageFont.truetype("/content/drive/MyDrive/Учеба/gamedev/НИР/Arial.ttf", 12, encoding="unic")
visualkeras.layered_view(model, legend=True, font=font)

filepath = modelFilePath
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1, save_best_only=True,mode='max')

log_dir = logsModel + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [checkpoint, tensorboard_callback]

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=10,
          verbose=1,
          callbacks=callbacks)

loaded_model = tf.keras.models.load_model(filepath)

test_loss, test_accuracy = loaded_model.evaluate(validation_generator)
print("Validation Loss: {}, Validation Accuracy: {}".format(test_loss, test_accuracy))

for images, labels in normalized_ds.take(1):
    for i in range(10):
        image = images[i]
        label = labels[i]
        image_batch = tf.expand_dims(image, [0])
        prediction = loaded_model.predict(image_batch)

        # Plotting
        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        plot_image(i, prediction[0], label, image)
        plt.subplot(1,2,2)
        plot_value_array(i, prediction[0],  label)
        plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np

for i in range(2):
  test_image_path = dir_test+str(i+1)+".jpg"  # Замените на фактический путь

# Загрузка и предобработка изображения
  img = image.load_img(test_image_path, target_size=(img_height, img_width))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0  # Нормализация значений пикселей

# Предсказание класса
  prediction = loaded_model.predict(img_array)

# Отображение результата
  plt.figure(figsize=(10, 3))
  plt.subplot(1, 2, 1)
  plt.imshow(img_array[0], cmap=plt.cm.binary)
  plt.subplot(1,2,2)
  plot_value_array(i, prediction[0],  label)
  plt.xlabel("{} {:2.0f}%".format(dataset[np.argmax(prediction)], 100 * np.max(prediction)))
  plt.show()

test_generator = train_datagen.flow_from_directory(
    dir_test,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=dataset,
    shuffle=True
)

test_loss, test_accuracy = loaded_model.evaluate(test_generator)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_accuracy))

images, labels = next(test_generator)
for i in range(4):
  image = images[i]
  label = labels[i]
  image_batch = tf.expand_dims(image, axis=0)
  prediction = loaded_model.predict(image_batch)

        # Plotting
  plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  plot_image(i, prediction[0], np.argmax(label), image)
  plt.subplot(1, 2, 2)
  plot_value_array(i, prediction[0], np.argmax(prediction[0]))
  plt.show()
