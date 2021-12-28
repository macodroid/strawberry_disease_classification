"""
Script load raw images and annotation.
Convert images in numpy array and parse label from JSON file and store it to numpy array as well and export to file.
Final format of file is .npy
"""
import json
import numpy as np
import cv2
import os
from sklearn import preprocessing


def _separate_files(type_of_data):
    json_files = []
    image_files = os.listdir(f'data/{type_of_data}/')  # -> return content of directory all files
    for file in image_files:
        if '.json' in file:
            json_files.append(file)
            image_files.remove(file)
    return image_files, json_files


def _load_images(images, type_of_data):
    img = []
    for image in images:
        img.append(cv2.imread(f'data/{type_of_data}/{image}', 1))
    return np.asarray(img)


def _load_target(jsons, type_of_data):
    target = []
    for tar in jsons:
        with open(f'data/{type_of_data}/{tar}') as handle:
            dump_json = json.loads(handle.read())
        target.append(dump_json['shapes'][0]['label'])
    return np.asarray(target)


def _categorize_target(labels):
    unique_category = np.unique(labels)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(unique_category)
    return encoder.transform(labels)


def get_data(type_of_data):
    imgs, jsons = _separate_files(type_of_data)
    imgs = _load_images(imgs, type_of_data)
    target = _categorize_target(_load_target(jsons, type_of_data))
    return imgs, target


if __name__ == '__main__':
    types_of_data = ['train', 'test', 'val']
    os.mkdir('dataset')
    for type_of_data in types_of_data:
        os.mkdir(f'dataset/{type_of_data}')
        X, y = get_data(type_of_data)
        np.save(f'dataset/{type_of_data}/features_{type_of_data}', X)
        np.save(f'dataset/{type_of_data}/labels_{type_of_data}', y)
