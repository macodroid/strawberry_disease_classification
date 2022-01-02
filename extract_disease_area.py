import os
import cv2
import numpy as np
import json

from numpy.core.fromnumeric import resize

from bouding_box import get_all_points


def extract_infected_area(image_name, json_file):
    image = cv2.imread(image_name)
    points = get_all_points(json_file)

    mask = np.zeros((image.shape[0], image.shape[1]))
    for p in points:
        cv2.fillConvexPoly(mask, np.asarray(p, dtype=np.int32), 1)

    mask = mask.astype(bool)

    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out


def main(image_name, image_json):

    # try:
    #     os.mkdir('extract_feature')
    # except:
    #     print('directory exists')
    #     os.rmdir('extract_feature')
    #     os.mkdir('extract_feature')

    out = extract_infected_area(image_name, image_json)
    dim = (235,235)
    resized_out = cv2.resize(out, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('Extracted Image', out)
    cv2.imshow('Resized Image', resized_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('out.jpg',resized_out)
    
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# img = 'data/train/anthracnose_fruit_rot66.jpg'
# img_json = 'data/train/anthracnose_fruit_rot66.json'
img = 'data/train/gray_mold119.jpg'
img_json = 'data/train/gray_mold119.json'

main(img, img_json)
