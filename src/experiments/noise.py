import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.experiments.algorithms import *
from src.experiments.metrics import distance_avg_metric
from src.experiments.utils import algorithm_to_color, img_from_file, normalize, algorithm_name
from src.main.python.algorithms.noise_image import gaussian_additive_noise, salt_and_pepper
from src.main.python.utils.ImageWrapper import ImageWrapper


def apply_salt_and_pepper(image, parameter):
    result = salt_and_pepper(ImageWrapper(Image.fromarray(image)), parameter, 1 - parameter, 1).draw_image()
    return np.array(result)


def apply_gaussian_noise(image, parameter):
    result = gaussian_additive_noise(ImageWrapper(Image.fromarray(image)), 0.0, parameter, 1).draw_image()
    return np.array(result)


def apply_experiment(algorithms, images, metric, transformation_func, transformation_parameters):
    transformations = [[transformation_func(img, parameter) for img in images] for parameter in
                       transformation_parameters]

    results = []
    for alg_index in range(len(algorithms)):
        algorithm = algorithms[alg_index]

        distances_avg_list = []
        times_global_list = []
        for parameter_transform in transformations:
            times_list = []
            for img_index in range(len(parameter_transform)):
                start = time.time()
                new_matches = algorithm(images[img_index], parameter_transform[img_index])
                times_list.append(time.time() - start)
                distances_avg_list.append(metric(new_matches))
            times_global_list.append(times_list)

        results.append(normalize(distances_avg_list))
        #results.append((normalize(distances_avg_list), times_global_list))

    return results


def algorithm_noise_test(algorithm, parameter, img, metric, apply_noise):
    h, w, _ = img.shape

    matches_list = []
    times_list = []
    for a in parameter:
        rotated_image = apply_noise(img, a)

        start = time.time()

        new_matches = algorithm(img, rotated_image)

        times_list.append(time.time() - start)

        matches_list.append(new_matches)

    return [metric(matches) for matches in matches_list], times_list


def run_gaussian_noise():
    img = img_from_file('../../images/Lenaclor.ppm')

    # algorithms = [surf, sift, kaze, akaze]
    algorithms = [kaze, akaze]

    parameter = np.arange(0, 50, 10)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        match, t = algorithm_noise_test(a, parameter, img, distance_avg_metric(), apply_gaussian_noise)
        print(match)
        match_results.append(match)
        time_results.append(t)

    fig = plt.figure()
    fig.suptitle('Ruido gaussiano')
    plt.xlabel('Sigma')
    plt.ylabel('Distancia promedio entre todos los matches')

    legends = []
    for r, i in zip(match_results, range(len(match_results))):
        plt.plot(parameter, normalize(r), algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

    fig = plt.figure()
    fig.suptitle('Ruido gaussiano')
    plt.xlabel('Sigma')
    plt.ylabel('Tiempo de procesamiento')

    legends = []
    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(parameter, t, algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)


def run_salt_and_pepper():
    img = img_from_file('../../images/Lenaclor.ppm')

    # algorithms = [surf, sift, kaze, akaze]
    algorithms = [kaze, akaze]

    parameter = np.arange(0, 0.1, 0.01)

    match_results = []
    time_results = []
    for a in algorithms:
        print("{0} / {1}".format(algorithms.index(a) + 1, len(algorithms)))
        match, t = algorithm_noise_test(a, parameter, img, distance_avg_metric(), apply_salt_and_pepper)
        print(match)
        match_results.append(match)
        time_results.append(t)

    fig = plt.figure()
    fig.suptitle('Ruido salt and pepper')
    plt.xlabel('Parametro')
    plt.ylabel('Distancia promedio entre todos los matches')

    legends = []
    for r, i in zip(match_results, range(len(match_results))):
        plt.plot(parameter, normalize(r), algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

    fig = plt.figure()
    fig.suptitle('Ruido salt and pepper')
    plt.xlabel('Parametro')
    plt.ylabel('Tiempo de procesamiento')

    legends = []
    for t, i in zip(time_results, range(len(time_results))):
        plt.plot(parameter, t, algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)


def run():
    images_path = ['../../images/Lenaclor.ppm']
    images = [img_from_file(path) for path in images_path]
    # algorithms = [surf, sift, kaze, akaze]
    algorithms = [kaze, akaze]

    parameters = np.arange(0, 30, 5)[1:]

    results = apply_experiment(
        algorithms=algorithms,
        images=images,
        metric=distance_avg_metric(),
        transformation_func=apply_salt_and_pepper,
        transformation_parameters=parameters
    )

    fig = plt.figure()
    fig.suptitle('Ruido gaussiano')
    plt.xlabel('Sigma')
    plt.ylabel('Distancia promedio entre todos los matches')

    legends = []
    for r, i in zip(results, range(len(results))):
        plt.plot(parameters, r, algorithm_to_color(algorithms[i]))
        legends.append(algorithm_name(algorithms[i]))

    plt.legend(legends)

    plt.show()


run()
