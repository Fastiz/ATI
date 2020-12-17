import numpy as np
import matplotlib.pyplot as plt
from src.experiments.algorithms import *
import time
import cv2

from src.experiments.metrics import distance_avg_metric
from src.experiments.utils import algorithm_to_color, img_from_file, normalize


def algorithm_scale_test(algorithm, percentages, img, metric):

    h, w, _ = img.shape

    matches_list = []
    times_list = []
    for p in percentages:
        scaled_img = cv2.resize(img, (int(h*p), int(w*p)))

        start = time.time()

        new_matches = algorithm(img, scaled_img)

        times_list.append(time.time() - start)

        matches_list.append(new_matches)

    return [metric(matches) for matches in matches_list], times_list


def run():
    img = img_from_file('../../images/Lenaclor.ppm')

    algorithms = [surf, sift, kaze, akaze]

    percentages = np.arange(0.5, 5, 0.1)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        match, t = algorithm_scale_test(a, percentages, img, distance_avg_metric())
        print(match)
        match_results.append(match)
        time_results.append(t)

    plt.figure()

    for r, i in zip(match_results, range(len(match_results))):
        plt.plot(percentages, normalize(r), algorithm_to_color(algorithms[i]))

    plt.figure()

    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(percentages, t, algorithm_to_color(algorithms[i]))

    plt.show()


run()
