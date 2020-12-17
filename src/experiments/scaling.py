import numpy as np
import matplotlib.pyplot as plt
from src.experiments.algorithms import *
import time
import cv2

from src.experiments.utils import algorithm_to_color, img_from_file


def best_n_metric(n):
    def metric(matches):
        return sum([m.distance for m in matches[0:n]])
    return metric


def threshold_metric(threshold):
    def metric(matches):
        return sum(map(lambda m: m.distance < threshold, matches))
    return metric


def distance_sum_metric():
    def metric(matches):
        return best_n_metric(len(matches))(matches)
    return metric


def distance_avg_metric():
    def metric(matches):
        return distance_sum_metric()(matches) / float(len(matches))
    return metric


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

        match_results.append(match)
        time_results.append(t)

    plt.figure()

    for r, i in zip(match_results, range(len(match_results))):
        plt.plot(percentages, r, algorithm_to_color(algorithms[i]))

    plt.figure()

    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(percentages, t, algorithm_to_color(algorithms[i]))

    plt.show()


run()
