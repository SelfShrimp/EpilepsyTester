import os

import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import TensorBoard
import datetime

img_height = 208
img_width = 176
logsModel = os.path.join(os.path.dirname(__file__), "logs/")

modelFilePath = os.path.join(os.path.dirname(__file__), 'models/')
loaded_model = tf.keras.models.load_model(modelFilePath)
def optimizeModel(count, train_generator, validation_generator):
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

    filepath = modelFilePath
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    log_dir = logsModel + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [checkpoint, tensorboard_callback]

    model.fit(train_generator,
              validation_data=validation_generator,
              epochs=1,
              verbose=1,
              callbacks=callbacks)

    loaded_model = tf.keras.models.load_model(filepath)
    test_loss, test_accuracy = loaded_model.evaluate(validation_generator)
    print("Validation Loss: {}, Validation Accuracy: {}".format(test_loss, test_accuracy))