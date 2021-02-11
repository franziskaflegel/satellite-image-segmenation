import numpy as np
import matplotlib.image as mpimg
import os
from PIL import Image as Im


def load_data(directory):
    images_orig = []
    labels_orig = []
    for entry in os.scandir(directory + '/images/'):
        if entry.path.endswith(".png") and entry.is_file():
            image = mpimg.imread(entry.path)
            image = image[:, :, :3]  # Drop the alpha channel
            label = mpimg.imread(entry.path.replace('images', 'labels'))
            label = np.expand_dims(label, axis=2)
            images_orig.append(image)
            labels_orig.append(label)

    images = np.array(images_orig)
    labels = np.array(labels_orig)

    assert (images.shape[3] == 3), "Input has wrong number of channels."
    assert (len(images.shape) == 4), "Input tensor has wrong shape."
    assert (len(labels.shape) == 4), "Label tensor has wrong shape."

    return images, labels


def load_test_data(directory):
    images_test_orig = []
    file_names = []
    for entry in os.scandir(directory):
        if entry.path.endswith(".png") and entry.is_file():
            file_name = entry.path.split("/")[-1].split('.')[0]
            file_names.append(file_name)
            image = mpimg.imread(entry.path)
            image = image[:, :, :3]  # Drop the alpha channel
            images_test_orig.append(image)

    images_test = np.array(images_test_orig)

    assert (images_test.shape[3] == 3), 'Input has wrong number of channels.'
    assert (len(images_test.shape) == 4), 'Input tensor has wrong shape.'

    return images_test, file_names


def save_test_predictions(predictions, directory, file_names):
    assert (predictions.shape[0] == len(file_names)), 'Length of file_names has to match first dimension of predictions'

    for i in range(predictions.shape[0]):
        prediction = predictions[i, :, :, 0]*255
        data = Im.fromarray(prediction).convert("L")
        data.save(directory + file_names[i] + '.png')
