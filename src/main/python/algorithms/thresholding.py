from typing import Tuple, Callable, List
import numpy as np
from src.main.python.utils.ImageWrapper import ImageWrapper


def calculate_pdf(channel: np.array) -> np.array:
    w, h = channel.shape

    count: int = 0
    result: np.array = np.zeros(256)
    for x in range(w):
        for y in range(h):
            val = channel[x, y]
            result[int(val)] += 1
            count += 1

    return np.array([val / count for val in result])


def calculate_cdf_from_pdf(pdf: np.array) -> np.array:
    result: np.array = np.zeros(256)

    prev_val: float = 0
    for t in range(256):
        result[t] = prev_val + pdf[t]
        prev_val = result[t]

    return result


def calculate_new_threshold(channel: np.array, condition: Callable[[float], bool]) -> float:
    w, h = channel.shape

    i1: int = 0
    i2: int = 0
    c1: int = 0
    c2: int = 0

    for x in range(w):
        for y in range(h):
            if condition(channel[x, y]):
                c1 += 1
                i1 += channel[x, y]
            else:
                c2 += 1
                i2 += channel[x, y]

    m1: float = i1/c1
    m2: float = i2/c2

    return 0.5 * (m1 + m2)


def negation_hoc(condition: Callable[[float], bool]) -> Callable[[float], bool]:
    def negation(x: float) -> bool:
        return not condition(x)
    return negation


def lower_than_hof(value: float) -> Callable[[float], bool]:
    def comparator(x) -> bool:
        return x < value
    return comparator


def filter_channel(channel: np.array, transformation: Callable[[float], float]) -> np.array:
    w, h = channel.shape
    new_channel = np.zeros(channel.shape)
    for x in range(w):
        for y in range(h):
            new_channel[x, y] = transformation(channel[x, y])
    return new_channel


def black_mask_hoc(condition: Callable[[float], bool]) -> Callable[[float], float]:
    def black_mask(x: float) -> float:
        if condition(x):
            return x
        return 0
    return black_mask


def get_regions(channel: np.array, t: float, mode: str) -> Tuple[ImageWrapper, ImageWrapper]:
    w, h = channel.shape

    region1: np.array = filter_channel(channel, black_mask_hoc(lower_than_hof(t)))
    image1: ImageWrapper = ImageWrapper.from_dimensions(w, h, mode)
    image1.channels = [region1]

    region2: np.array = filter_channel(channel, black_mask_hoc(negation_hoc(lower_than_hof(t))))
    image2: ImageWrapper = ImageWrapper.from_dimensions(w, h, mode)
    image2.channels = [region2]

    return image1, image2


def global_thresholding(image: ImageWrapper, starting_t: float, epsilon: float) -> Tuple[ImageWrapper, ImageWrapper, int, int]:
    if len(image.channels) > 1:
        raise ValueError("Method only supports one channel")
    channel = image.channels[0]

    iteration_count: int = 0
    prev_t: float = float('inf')
    curr_t: float = starting_t
    while abs(prev_t - curr_t) > epsilon:
        iteration_count += 1
        prev_t = curr_t
        curr_t = calculate_new_threshold(channel, lower_than_hof(curr_t))

    image1, image2 = get_regions(channel, curr_t, image.get_mode())

    return image1, image2, int(curr_t), iteration_count


def otsu_method(image: ImageWrapper) -> Tuple[ImageWrapper, ImageWrapper, List[int]]:
    results = []
    for channel in image.channels:
        pdf: np.array = calculate_pdf(channel)
        cdf: np.array = calculate_cdf_from_pdf(pdf)

        accum_means: np.array = calculate_cdf_from_pdf(np.array([i * p for i, p in zip(range(256), pdf)]))
        global_mean: float = accum_means[-1]

        variance = []
        for t in range(256):
            nominator = np.power(global_mean * cdf[t] - accum_means[t], 2)
            denominator = cdf[t]*(1-cdf[t])
            variance.append(nominator/denominator if not np.isclose(denominator, 0) else 0)
        variance = np.array(variance)

        arg_max = np.argwhere(variance == max(variance))

        t = sum(arg_max) / len(arg_max)

        image1, image2 = get_regions(channel, t, image.get_mode())

        results.append([image1, image2, int(t)])

    image1_channels = []
    image2_channels = []
    ts = []
    for result in results:
        image1, image2, t = result
        image1_channels.append(image1.channels[0])
        image2_channels.append(image2.channels[0])
        ts.append(t)

    image1, image2, _ = results[0]

    image1.channels = image1_channels
    image2.channels = image2_channels

    return image1, image2, ts
