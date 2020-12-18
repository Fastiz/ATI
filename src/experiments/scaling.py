import numpy as np
import matplotlib.pyplot as plt
from src.experiments.algorithms import *
import time
import cv2

from src.experiments.metrics import distance_avg_metric
from src.experiments.utils import algorithm_to_color, img_from_file, normalize, algorithm_name, scalar_multiplication, \
    sum_lists


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
    imgs = [img_from_file('../../dataset/img' + str(i) + '.ppm') for i in range(1, 7)]

    algorithms = [surf, sift, kaze, akaze]

    percentages = np.arange(0.5, 5, 0.25)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        sum_list_error = [0] * len(percentages)
        sum_list_time = [0] * len(percentages)
        for img in imgs:
            match, t = algorithm_scale_test(a, percentages, img, distance_avg_metric())

            sum_list_error = sum_lists(sum_list_error, match)
            sum_list_time = sum_lists(sum_list_time, t)

        match_results.append(scalar_multiplication(1.0 / len(imgs), sum_list_error))
        time_results.append(scalar_multiplication(1.0 / len(imgs), sum_list_time))

    fig = plt.figure()
    fig.suptitle('Escalado')
    plt.xlabel('Porcentaje')
    plt.ylabel('Distancia promedio entre todos los matches')

    legends = []
    for r, i in zip(match_results, range(len(match_results))):
        plt.plot(percentages, normalize(r), algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

    for r, i in zip(match_results, range(len(match_results))):
        normalized_r = normalize(r)
        plt.axhline(y=sum(normalized_r) / len(normalized_r), color=algorithm_to_color(algorithms[i]), linestyle='-')

    fig = plt.figure()
    fig.suptitle('Escalado')
    plt.xlabel('Parametro')
    plt.ylabel('Tiempo de procesamiento')

    legends = []
    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(percentages, t, algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

run()
plt.show()