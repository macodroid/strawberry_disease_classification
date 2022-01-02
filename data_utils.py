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


def separate_files(path, data_type):
    json_files = []
    image_files = []
    data_in_dir = os.listdir(f'{path}/{data_type}')  # -> return content of directory all files
    # I don't know why but when running on Windows os.lastdir return content of directory in
    # same order as in actual directory but running on linux it returns in files in random position. WHY Linux ???
    data_in_dir.sort()
    for file in data_in_dir:
        if ".json" in file:
            json_files.append(file)
        else:
            image_files.append(file)
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


# if __name__ == '__main__':
#     types_of_data = ['train', 'test', 'val']
#     os.mkdir('dataset')
#     for type_of_data in types_of_data:
#         images, labels = get_data(type_of_data)
#         separate_images(images, type_of_data)
#         data_arr = np.append(images, labels, axis=1)
#         np.savetxt(f'dataset/data_{type_of_data}.csv', data_arr, delimiter=',', fmt='%s')
if __name__ == '__main__':
    types_of_data = ['train', 'test', 'val']
    path_to_json = 'data'
    path_to_images = 'extract_dataset'
    for d_type in types_of_data:
        images, labels = separate_files(path_to_json, d_type)
        extract_images, dont_care = separate_files(path_to_images, d_type)
        target = categorize_target(load_target(labels, d_type))
        extract_images = np.reshape(np.asarray(extract_images), (-1, 1))
        target = np.reshape(np.asarray(target), (-1, 1))
        data_arr = np.append(extract_images, target, axis=1)
        np.savetxt(f'{path_to_images}/extracted_data_{d_type}.csv', data_arr, delimiter=',', fmt='%s')
