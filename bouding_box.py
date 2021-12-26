import numpy as np
import cv2 as cv
import json


def get_points(json_file_path):
    with open(json_file_path) as handle:
        dictdump = json.loads(handle.read())
    return np.asarray(dictdump['shapes'][0]['points'], np.int32)


if __name__ == '__main__':
    path_image = 'data/train/powdery_mildew_leaf517.jpg'
    path_image_json = 'data/train/powdery_mildew_leaf517.json'

    bounding_box_points = get_points(path_image_json)

    img = cv.imread(path_image)
    cv.polylines(img, [bounding_box_points], isClosed=True, color=(255, 0, 0), thickness=2)

    cv.imshow("Strawberry", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

