"""
Scripts loads images and jsons file mixed in one directory.
Separate the images in specific directories. Extract labels from JSON file.
Create CSV file with name of image and corresponding label (target variable).
"""

import json
import numpy as np
import cv2
import os
import shutil

from sklearn import preprocessing


def separate_files(data_type):
    json_files = []
    image_files = os.listdir(f'data/{data_type}/')  # -> return content of directory all files
    for file in image_files:
        if '.json' in file:
            json_files.append(file)
            image_files.remove(file)
    return image_files, json_files


def separate_images(_images, data_type):
    os.mkdir(f'dataset/{data_type}')
    for image in _images:
        shutil.copy(f'data/{data_type}/{image[0]}', f'dataset/{data_type}/{image[0]}')


def _load_images(_images, data_type):
    img = []
    for image in _images:
        img.append(cv2.imread(f'data/{data_type}/{image}', 1))
    return np.asarray(img)


def load_target(jsons, data_type):
    target = []
    for tar in jsons:
        with open(f'data/{data_type}/{tar}') as handle:
            dump_json = json.loads(handle.read())
        target.append(dump_json['shapes'][0]['label'])
    return np.asarray(target)


def categorize_target(target_var):
    unique_category = np.unique(target_var)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(unique_category)
    return encoder.transform(target_var)


# def get_data(data_type):
#     images, jsons = separate_files(data_type)
#     images = _load_images(images, data_type)
#     target = categorize_target(load_target(jsons, data_type))
#     return images, target

def get_data(data_type):
    _images, jsons = separate_files(data_type)
    target = categorize_target(load_target(jsons, data_type))
    return np.reshape(np.asarray(_images), (-1, 1)), np.reshape(np.asarray(target), (-1, 1))


if __name__ == '__main__':
    types_of_data = ['train', 'test', 'val']
    os.mkdir('dataset')
    for type_of_data in types_of_data:
        images, labels = get_data(type_of_data)
        separate_images(images, type_of_data)
        data_arr = np.append(images, labels, axis=1)
        np.savetxt(f'dataset/data_{type_of_data}.csv', data_arr, delimiter=',', fmt='%s')