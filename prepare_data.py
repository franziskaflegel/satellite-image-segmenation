import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented

plt.ion()


def load_data(directory):
    X_orig = []
    Y_orig = []
    for entry in os.scandir(directory):
        if entry.path.endswith(".png") and entry.is_file():
            X = mpimg.imread(entry.path)
            X = X[:, :, :3]  # Drop the alpha channel
            Y = mpimg.imread(entry.path.replace('images', 'labels'))
            X_orig.append(X)
            Y_orig.append(Y)

    imgs_np = np.array(X_orig)
    masks_np = np.array(Y_orig)

    assert (imgs_np.shape[3] == 3), "Input has wrong number of channels."
    assert (len(imgs_np.shape) == 4), "Input tensor has wrong shape."
    assert (len(masks_np.shape) == 3), "Label tensor has wrong shape."

    return imgs_np, masks_np

def load_test_data(directory):
    X_test = []
    for entry in os.scandir(directory):
        if entry.path.endswith(".png") and entry.is_file():
            X = mpimg.imread(entry.path)
            X = X[:, :, :3]  # Drop the alpha channel
            X_test.append(X)

    test_np = np.array(X_test)

    assert (test_np.shape[3] == 3), "Input has wrong number of channels."
    assert (len(test_np.shape) == 4), "Input tensor has wrong shape."

    return test_np


def train_val_split(x, y, val_ratio):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio, random_state=0)
    y_train = np.expand_dims(y_train, axis=3)
    y_val = np.expand_dims(y_val, axis=3)
    
    return x_train, x_val, y_train, y_val

def get_train_set_augmented(x_train, y_train, batch_size, rotation_range, channel_shift_range, zoom_range):
    train_gen = get_augmented(
        x_train, y_train, batch_size=batch_size,
        data_gen_args = dict(
            rotation_range=rotation_range,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0,
            channel_shift_range=channel_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant'
        ))

    return train_gen