import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.experiments.algorithms import *
from src.experiments.metrics import distance_avg_metric
from src.experiments.utils import algorithm_to_color, img_from_file, normalize, algorithm_name


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
    img = img_from_file('../../images/Lenaclor.ppm')

    algorithms = [surf, sift, kaze, akaze]
    #algorithms = [kaze, akaze]

    angles = np.arange(0, 360, 90)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        match, t = algorithm_rotation_test(a, angles, img, distance_avg_metric())
        print(match)
        match_results.append(match)
        time_results.append(t)

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