import numpy as np

from src.main.python.algorithms.operations_between_images import pixel_transformation
from src.main.python.utils.ImageWrapper import ImageWrapper


def multiplicative_noise(random_generator):
    def noise(x, y, val):
        rnd = random_generator()
        return [random_generator() * v for v in val]
    return noise


def additive_noise(random_generator):
    def noise(x, y, val):
        result = [random_generator() + v for v in val]
        return result
    return noise


def gaussian_additive_noise(image: ImageWrapper, mu: float, sigma: float):
    return pixel_transformation(image, additive_noise(lambda: np.random.normal(loc=mu, scale=sigma)))


def rayleigh_multiplicative_noise(image: ImageWrapper, gamma):
    return pixel_transformation(image, multiplicative_noise(lambda: np.random.rayleigh(scale=gamma)))


def exponential_multiplicative_noise(image: ImageWrapper, _lambda):
    return pixel_transformation(image, multiplicative_noise(lambda: np.random.exponential(scale=1/_lambda)))


def salt_and_pepper(image: ImageWrapper, p0: float, p1: float):
    def noise(x, y, val):
        res = []

        for v in val:
            rnd: float = np.random.uniform(0, 1)

            if rnd < p0:
                res.append(0)
            elif rnd > p1:
                res.append(1)
            else:
                res.append(v)

        return res

    return pixel_transformation(image, noise)
