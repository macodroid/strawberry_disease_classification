import os
import cv2
import numpy as np
import shutil

from bouding_box import get_all_points
from data_utils import separate_files


def extract_infected_area(image_name, json_file, path_to_file):
    image = cv2.imread(f'{path_to_file}/{image_name}')
    points = get_all_points(f'{path_to_file}/{json_file}')

    mask = np.zeros((image.shape[0], image.shape[1]))
    for p in points:
        cv2.fillConvexPoly(mask, np.asarray(p, dtype=np.int32), 1)

    mask = mask.astype(bool)

    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out


def main():
    raw_data = 'data'
    root_dir = 'extract_dataset'
    data_type = ['train', 'test', 'val']
    try:
        os.mkdir(root_dir)
    except:
        print('directory exists')
        shutil.rmtree(root_dir)
        os.mkdir(root_dir)

    for d_type in data_type:
        path_to_save = f'{root_dir}/{d_type}'
        path_to_file = f'{raw_data}/{d_type}'
        os.mkdir(path_to_save)
        image_name, json_name = separate_files(d_type)
        for i in range(len(image_name)):
            extracted_disease_img = extract_infected_area(image_name[i], json_name[i], path_to_file)
            cv2.imwrite(f'{path_to_save}/extract_{image_name[i]}', extracted_disease_img)


main()
