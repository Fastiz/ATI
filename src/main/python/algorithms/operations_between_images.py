import numpy as np

from src.main.python.algorithms.channel_operations import channel_histogram
from src.main.python.utils.ImageWrapper import ImageWrapper


def product(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: (a * b) % 255)


def subtraction(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: abs((a - b) % 255))


def addition(first_image: ImageWrapper, second_image: ImageWrapper):
    return pixel_to_pixel_operation(first_image, second_image, lambda a, b: (a + b) % 255)


def scalar_multiplication(first_image: ImageWrapper, scalar: float):
    w, h = first_image.dimensions()

    result = ImageWrapper.from_dimensions(w, h)

    for x in range(w):
        for y in range(h):
            pixel = first_image.get_pixel(x, y)
            val = [int(first_image.get_pixel(x, y)[i] * scalar) % 255 for i in range(len(first_image.get_pixel(x, y)))]
            result.set_pixel(x, y, tuple(val))

    return result


def pixel_to_pixel_operation(first_image: ImageWrapper, second_image: ImageWrapper, operation):
    assert first_image.dimensions() == second_image.dimensions()

    w, h = first_image.dimensions()

    result = ImageWrapper.from_dimensions(w, h)

    for x in range(w):
        for y in range(h):
            val = [operation(first_image.get_pixel(x, y)[i], second_image.get_pixel(x, y)[i])
                   for i in range(len(first_image.get_pixel(x, y)))]

            result.set_pixel(x, y, tuple(val))

    return result


def pixel_transformation(image: ImageWrapper, transformation_function):
    w, h = image.dimensions()

    result = ImageWrapper.from_dimensions(w, h, mode=image.get_mode())

    for x in range(w):
        for y in range(h):
            val = transformation_function(x, y, image.get_pixel(x, y))

            result.set_pixel(x, y, tuple(val))

    return result


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

        h, w = channel.size

        for x in range(w):
            for y in range(h):
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
