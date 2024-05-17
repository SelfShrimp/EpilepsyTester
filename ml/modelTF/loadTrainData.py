import os

from ml.modelTF.optimizeModel import optimizeModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dir  = os.path.join(os.path.dirname(__file__), "train/")
font = os.path.join(os.path.dirname(__file__), "Arial.ttf")
dir_test = os.path.join(os.path.dirname(__file__), "test/")

logsModel = os.path.join(os.path.dirname(__file__), "logs/")

modelFilePath = os.path.join(os.path.dirname(__file__), 'models/')

img_height = 208
img_width = 176

#print(count)
#print(dataset)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size= 16
data_dir = dir

dataset = [f.name for f in os.scandir(dir) if f.is_dir()]


def trainModel():

    count = len(dataset)
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

    optimizeModel(count, train_generator, validation_generator)

