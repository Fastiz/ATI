import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.experiments.algorithms import *


# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
from src.experiments.utils import algorithm_to_color, img_from_file


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def algorithm_rotation_test(algorithm, angle, img, threshold):
    h, w, _ = img.shape

    matches_list = []
    times_list = []
    for a in angle:
        rotated_image = rotate_image(img, a)

        start = time.time()

        new_matches = algorithm(img, rotated_image)

        times_list.append(time.time() - start)

        matches_list.append(new_matches)

    return [sum(map(lambda m: m.distance < threshold, matches)) for matches in matches_list], times_list


def run():
    img = img_from_file('../../images/Lenaclor.ppm')

    algorithms = [surf, sift, kaze, akaze]

    angles = np.arange(0, 360, 45)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        match, t = algorithm_rotation_test(a, angles, img, 500)

        match_results.append(match)
        time_results.append(t)

    plt.figure()

    for r, i in zip(match_results, range(len(match_results))):
        plt.plot(angles, r, algorithm_to_color(algorithms[i]))

    plt.figure()

    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(angles, t, algorithm_to_color(algorithms[i]))

    plt.show()


run()