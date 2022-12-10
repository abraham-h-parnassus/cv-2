import math

import cv2
import numpy as np

DEPTH = 3


def c(value):
    return int(math.ceil(value))


def affine_transform(image_path, params):
    image = cv2.imread(image_path)
    return _do_transform(image, params)


def _do_transform(image, params):
    transformation = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0, 0, 1]
    ])

    max_x = 0
    max_y = 0
    result = np.empty((2000, 2000, 3), dtype=np.uint8)
    for x, row in enumerate(image):
        for y, col in enumerate(row):
            pixel_data = image[x, y, :]
            input_coords = np.array([x, y, 1])
            x_out, y_out, _ = transformation @ input_coords
            result[c(x_out), c(y_out), :] = pixel_data
            if x_out > max_x:
                max_x = c(x_out)
            if y_out > max_y:
                max_y = c(y_out)

    return result[0:max_x, 0:max_y, :]


def normalize(reference_points, transformed_image, transformed_points):
    if len(reference_points) != 3 or len(transformed_points) != 3:
        print("Select reference points by clicking on the images")
        exit(1)
    x1, x2, x3 = [p[0] for p in reference_points]
    y1, y2, y3 = [p[1] for p in reference_points]

    x1t, x2t, x3t = [p[0] for p in transformed_points]
    y1t, y2t, y3t = [p[1] for p in transformed_points]

    a11 = ((x1t - x2t) * (y2 - y3) - (x2t - x3t) * (y1 - y2)) / ((x3 - x2) * (y1 - y2) - (x2 - x1) * (y2 - y3))
    a12 = ((x1t - x2t) * (x2 - x3) - (x2t - x3t) * (x1 - x2)) / ((y3 - y2) * (x1 - x2) - (y2 - y1) * (x2 - x3))
    a13 = x1t \
          - ((x1t - x2t) * (y2 - y3) * x1 - (x2t - x3t) * (y1 - y2) * x1) / (
                  (x3 - x2) * (y1 - y2) - (x2 - x1) * (y2 - y3)) \
          - ((x1t - x2t) * (x2 - x3) * y1 - (x2t - x3t) * (x1 - x2) * y1) / (
                  (y3 - y2) * (x1 - x2) - (y2 - y1) * (x2 - x3))
    a21 = ((y1t - y2t) * (y2 - y3) - (y2t - y3t) * (y1 - y2)) / ((x3 - x2) * (y1 - y2) - (x2 - x1) * (y2 - y3))

    a22 = ((y1t - y2t) * (x2 - x3) - (y2t - y3t) * (x1 - x2)) / ((y3 - y2) * (x1 - x2) - (y2 - y1) * (x2 - x3))

    a23 = y1t \
          - ((y1t - y2t) * (y2 - y3) * x1 - (y2t - y3t) * (y1 - y2) * x1) / (
                  (x3 - x2) * (y1 - y2) - (x2 - x1) * (y2 - y3)) \
          - ((y1t - y2t) * (x2 - x3) * y1 - (y2t - y3t) * (x1 - x2) * y1) / (
                  (y3 - y2) * (x1 - x2) - (y2 - y1) * (x2 - x3))
    T = np.array([
        [a11, a12, a13],
        [a21, a22, a23],
        [0, 0, 1]
    ])
    RT = np.linalg.inv(T)
    return _do_transform(transformed_image, [RT[0][0], RT[0][1], RT[0][2], RT[1, 0], RT[1, 1], RT[1, 2]])
