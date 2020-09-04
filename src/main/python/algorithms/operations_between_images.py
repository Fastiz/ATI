import numpy as np
from typing import List, Tuple

from src.main.python.algorithms.channel_operations import channel_histogram
from src.main.python.utils.ImageWrapper import ImageWrapper


def product(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: (a * b))


def subtraction(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: abs((a - b)))


def addition(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: (a + b))


def scalar_multiplication(first_image: ImageWrapper, scalar: float):
    w, h = first_image.dimensions()

    result = ImageWrapper.from_dimensions(w, h, mode=first_image.get_mode())

    for x in range(w):
        for y in range(h):
            val = [int(first_image.get_pixel(x, y)[i] * scalar) % 255 for i in range(len(first_image.get_pixel(x, y)))]
            result.set_pixel(x, y, tuple(val))

    return result


def matrix_to_image(matrix: List[List[tuple]], mode=None) -> ImageWrapper:
    w: int = len(matrix)
    h: int = len(matrix[0])
    result: ImageWrapper = ImageWrapper.from_dimensions(w, h, mode)
    for x in range(w):
        for y in range(h):
            result.set_pixel(x, y, tuple([int(a) for a in matrix[x][y]]))
    return result


def normalized_matrix(matrix: List[List[tuple]]) -> List[List[tuple]]:
    w: int = len(matrix)
    h: int = len(matrix[0])

    max_values: List[float] = [-float("inf")]*len(matrix[0][0])
    min_values: List[float] = [float("inf")]*len(matrix[0][0])
    for x in range(w):
        for y in range(h):
            val = matrix[x][y]
            for i in range(len(val)):
                v = val[i]
                if max_values[i] < v:
                    max_values[i] = v
                if min_values[i] > v:
                    min_values[i] = v

    new_matrix: List[List[tuple]] = [[()]*h for i in range(w)]
    for x in range(w):
        for y in range(h):
            val = matrix[x][y]
            new_matrix[x][y] = tuple([255 * (val[i] - min_values[i]) / (max_values[i] - min_values[i])
                                      if max_values[i] != min_values[i] else val[i]
                                      for i in range(len(val))])

    return new_matrix


def pixel_to_pixel_operation(first_image: ImageWrapper, second_image: ImageWrapper, operation):
    assert first_image.dimensions() == second_image.dimensions()

    w, h = first_image.dimensions()
    matrix = []

    for x in range(w):
        row = []
        for y in range(h):
            val = [operation(first_image.get_pixel(x, y)[i], second_image.get_pixel(x, y)[i])
                   for i in range(len(first_image.get_pixel(x, y)))]
            row.append(tuple(val))
        matrix.append(row)

    return matrix_to_image(normalized_matrix(matrix), mode=first_image.get_mode())


def pixel_transformation(image: ImageWrapper, transformation_function):
    w, h = image.dimensions()

    matrix = []

    for x in range(w):
        row = []
        for y in range(h):
            val = transformation_function(x, y, image.get_pixel(x, y))

            row.append(tuple(val))
        matrix.append(row)

    return matrix_to_image(normalized_matrix(matrix), mode=image.get_mode())


def equalize_histogram(image: ImageWrapper):
    channels = image.get_image_element().split()
    eq_channel_mappings = []

    for channel in channels:
        channel_hist = channel_histogram(channel, display=True)

        eq_channel_mapping_buffer = [0] * len(channel_hist)
        for k in range(len(channel_hist)):
            for j in range(k):
                eq_channel_mapping_buffer[k] += channel_hist[j]

        eq_channel_mapping = []
        s_min = min(eq_channel_mapping_buffer)
        for k in range(len(channel_hist)):
            eq_channel_mapping.append(round(255 * ((eq_channel_mapping_buffer[k] - s_min) / (1 - s_min))))

        eq_channel_mappings.append(eq_channel_mapping)

    def mapping_function(x, y, value):
        return [eq_channel_mappings[i][value[i]] for i in range(len(value))]

    result = pixel_transformation(image, mapping_function)

    for channel in result.image_element.split():
        channel_histogram(channel, display=True)

    return result


def max_gray_value(image: ImageWrapper):
    channels = image.image_element.split()

    max_values = [0] * len(channels)
    for i in range(len(channels)):
        channel = channels[i]

        w, h = channel.size

        for x in range(w-1):
            for y in range(h-1):
                p = channel.getpixel((x, y))
                if max_values[i] < p:
                    max_values[i] = p

    return max_values


def dynamic_range_compression(image: ImageWrapper):
    max_values = max_gray_value(image)

    c = [255 / np.log(1 + R) for R in max_values]

    def transformation(x, y, val):
        return [int(c[i] * np.log(1 + val[i])) for i in range(len(val))]

    return pixel_transformation(image, transformation)


def gamma_power_function(image: ImageWrapper, gamma: float):
    c = np.power(255, 1-gamma)

    def transformation(x, y, val):
        return [int(c * np.power(r, gamma)) for r in val]

    return pixel_transformation(image, transformation)
