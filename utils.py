import json
import numpy as np
import cv2
import os
from sklearn import preprocessing


def _separate_files():
    json_files = []
    image_files = os.listdir('data/train/')  # -> return content of directory all files
    for file in image_files:
        if '.json' in file:
            json_files.append(file)
            image_files.remove(file)
    return image_files, json_files


def load_images(images):
    img = []
    for image in images:
        img.append(cv2.imread('data/train/{0}'.format(image), 1))
    return np.asarray(img)


def load_target(jsons):
    target = []
    for tar in jsons:
        with open('data/train/{0}'.format(tar)) as handle:
            dump_json = json.loads(handle.read())
        target.append(dump_json['shapes'][0]['label'])
    return np.asarray(target)


def categorize_target(labels):
    unique_category = np.unique(labels)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(unique_category)
    return encoder.transform(labels)


images, jsons = _separate_files()
d = load_images(images)
label = categorize_target(load_target(jsons))
print('d')
