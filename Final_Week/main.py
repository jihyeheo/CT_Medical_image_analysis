from lenet import LeNet
from load_dataset import Load_data
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from glob import glob
import random
import shutil

weightsPath = "weights/lenet_weights.hdf5"

print("[INFO] Making dataset ...")

BASE_DIR = os.getcwd()
data_dic, num = Load_data.load_data_files(BASE_DIR)
Load_data.train_validation_split(BASE_DIR, data_dic, split_ratio=0.2)

batch_size = 128
num_classes = 3
epochs = 100

preprocessing_image = tf.keras.preprocessing.image

train_datagen = preprocessing_image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "split_dataset/train"),
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical', color_mode="grayscale")

validation_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "split_dataset/validation"),
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical', color_mode="grayscale")

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build((32,32,1), classes = 3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training...")
model.fit_generator(train_generator, steps_per_epoch=num//batch_size, epochs = epochs, validation_data=validation_generator, validation_steps=1)

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate_generator(validation_generator, steps = 5)
    
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

print("[INFO] dumping weights to file...")
model.save_weights(weightsPath, overwrite=True)
