from src.experiments.algorithms import *
import numpy as np


def algorithm_to_color(algorithm):
    dic = {
        surf: 'g',
        sift: 'b',
        kaze: 'r',
        akaze: 'y'
    }

    return dic[algorithm]


def algorithm_name(algorithm):
    dic = {
        surf: 'SURF',
        sift: 'SIFT',
        kaze: 'KAZE',
        akaze: 'AKAZE'
    }

    return dic[algorithm]


def img_from_file(path):
    return cv2.imread(path)


def normalize(values):
    max_value = max(values)
    min_value = min(values)

    return [(x - min_value) / (max_value - min_value) for x in values]