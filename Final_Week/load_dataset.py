from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from glob import glob
import random
import shutil

class Load_data : 
    def load_data_files(base_dir):
        folder_name = "dataset"
        RAW_DATASET = os.path.join(base_dir, folder_name)

        abs_dir = os.path.join(os.getcwd(), folder_name)
        sub_dir = os.listdir(abs_dir)
        data_dic = {}
        
        for class_name  in sub_dir:
            imgs = glob(os.path.join(RAW_DATASET,class_name,"*.jpg"))

            data_dic[class_name] = imgs
            print("Class: {}".format(class_name))
            print("Number of images: {} \n".format(len(imgs)))

        return data_dic, len(imgs)

    def copy_files_to_directory(files, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Created directory: {}".format(directory))

        for f in files:
            shutil.copy(f, directory)
        print("Copied {} files.\n".format(len(files)))

    def train_validation_split(base_dir, data_dic, split_ratio=0.2):
        DATASET = os.path.join(base_dir,"split_dataset")

        if not os.path.exists(DATASET):
            os.makedirs(DATASET)

        for class_name, imgs in data_dic.items():
            idx_split = int(len(imgs) * split_ratio)
            random.shuffle(imgs)
            validation = imgs[:idx_split]
            train = imgs[idx_split:]

            Load_data.copy_files_to_directory(train, os.path.join(DATASET,"train",class_name))
            Load_data.copy_files_to_directory(validation, os.path.join(DATASET,"validation",class_name))


