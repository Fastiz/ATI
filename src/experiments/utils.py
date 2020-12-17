from src.experiments.algorithms import *


def algorithm_to_color(algorithm):
    dic = {
        surf: 'g',
        sift: 'b',
        kaze: 'r',
        akaze: 'y'
    }

    return dic[algorithm]


def img_from_file(path):
    return cv2.imread(path)
