import numpy as np
import cv2 as cv
import json


# get just one bounding box
def get_points(json_file_path):
    with open(json_file_path) as handle:
        dictdump = json.loads(handle.read())
    return np.asarray(dictdump['shapes'][0]['points'], np.int32)


# get all bounding boxes
def get_all_points(json_file_path):
    points = []
    with open(json_file_path) as handle:
        dictdump = json.loads(handle.read())
    boxes = dictdump['shapes']
    for box_points in boxes:
        points.append(box_points['points'])
    return points


if __name__ == '__main__':
    path_image = 'data/train/angular_leafspot1.jpg'
    path_image_json = 'data/train/angular_leafspot1.json'

    bounding_box_points = get_all_points(path_image_json)

    img = cv.imread(path_image)
    for points in bounding_box_points:
        cv.polylines(img, [np.asarray(points, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    cv.imshow("Strawberry", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
