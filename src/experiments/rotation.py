import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.experiments.algorithms import *
from src.experiments.metrics import distance_avg_metric
from src.experiments.utils import algorithm_to_color, img_from_file, normalize, algorithm_name, sum_lists, \
    scalar_multiplication


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def algorithm_rotation_test(algorithm, angle, img, metric):
    h, w, _ = img.shape

    matches_list = []
    times_list = []
    for a in angle:
        rotated_image = rotate_image(img, a)

        start = time.time()

        new_matches = algorithm(img, rotated_image)

        times_list.append(time.time() - start)

        matches_list.append(new_matches)

    return [metric(matches) for matches in matches_list], times_list


def run():
    imgs = [img_from_file('../../dataset/img' + str(i) + '.ppm') for i in range(1, 7)]

    algorithms = [surf, sift, kaze, akaze]
    #algorithms = [kaze, akaze]

    angles = np.arange(0, 360, 10)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        sum_list_error = [0] * len(angles)
        sum_list_time = [0] * len(angles)
        for img in imgs:
            match, t = algorithm_rotation_test(a, angles, img, distance_avg_metric())

            sum_list_error = sum_lists(sum_list_error, match)
            sum_list_time = sum_lists(sum_list_time, t)

        match_results.append(scalar_multiplication(1.0/len(imgs), sum_list_error))
        time_results.append(scalar_multiplication(1.0/len(imgs), sum_list_time))

    fig = plt.figure()
    fig.suptitle('Rotación')
    plt.xlabel('Angulo de rotación (grados)')
    plt.ylabel('Distancia promedio entre todos los matches')

    legends = []
    for r, i in zip(match_results, range(len(match_results))):
        normalized_r = normalize(r)
        plt.plot(angles, normalized_r, algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

    for r, i in zip(match_results, range(len(match_results))):
        normalized_r = normalize(r)
        plt.axhline(y=sum(normalized_r) / len(normalized_r), color=algorithm_to_color(algorithms[i]), linestyle='-')

    fig = plt.figure()
    fig.suptitle('Rotación')
    plt.xlabel('Angulo de rotación (grados)')
    plt.ylabel('Tiempo de procesamiento')

    legends = []
    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(angles, t, algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

run()
plt.show()