from random import sample, random

import numpy as np

from src.main.python.algorithms.operations_between_images import pixel_transformation
from src.main.python.utils.ImageWrapper import ImageWrapper


def multiplicative_noise(random_generator):
    def noise(x, y, val):
        return [random_generator() * v for v in val]

    return noise


def additive_noise(random_generator):
    def noise(x, y, val):
        result = [random_generator() + v for v in val]
        return result

    return noise


def noise_percentage(noise_fn, percentage: float, w: int, h: int):
    # coords = [(a, b) for a in range(w) for b in range(h)]
    # sample_list = sample(coords, int(w * h * percentage))
    #
    # def noise_wrapper(x, y, val):
    #     if (x, y) in sample_list:
    #         sample_list.remove((x, y))
    #         return noise_fn(x, y, val)
    #     return val

    def noise_wrapper(x, y, val):
        if random() < percentage:
            return noise_fn(x, y, val)

        return val

    return noise_wrapper


def gaussian_additive_noise(image: ImageWrapper, mu: float, sigma: float, percentage: float):
    w, h = image.dimensions()
    fn = noise_percentage(additive_noise(lambda: np.random.normal(loc=mu, scale=sigma)), percentage, w, h)
    return pixel_transformation(image, fn)


def rayleigh_multiplicative_noise(image: ImageWrapper, gamma, percentage: float):
    w, h = image.dimensions()
    fn = noise_percentage(multiplicative_noise(lambda: np.random.rayleigh(scale=gamma)), percentage, w, h)
    return pixel_transformation(image, fn)


def exponential_multiplicative_noise(image: ImageWrapper, _lambda, percentage: float):
    w, h = image.dimensions()
    fn = noise_percentage(multiplicative_noise(lambda: np.random.exponential(scale=1 / _lambda)), percentage, w, h)
    return pixel_transformation(image, fn)


def salt_and_pepper(image: ImageWrapper, p0: float, p1: float, percentage: float):
    def noise(x, y, val):
        res = []

        for v in val:
            rnd: float = np.random.uniform(0, 1)

            if rnd < p0:
                res.append(0)
            elif rnd > p1:
                res.append(255)
            else:
                res.append(v)

        return res

    w, h = image.dimensions()
    fn = noise_percentage(noise, percentage, w, h)
    return pixel_transformation(image, fn)
