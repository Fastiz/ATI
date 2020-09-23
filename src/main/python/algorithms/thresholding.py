from typing import Tuple, Callable
import numpy as np
from src.main.python.utils.ImageWrapper import ImageWrapper


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


def global_thresholding(image: ImageWrapper, starting_t: float, epsilon: float) -> Tuple[ImageWrapper, ImageWrapper, int, int]:
    if len(image.channels) > 1:
        raise ValueError("Method only supports one channel")
    channel = image.channels[0]
    w, h = channel.shape

    iteration_count: int = 0
    prev_t: float = float('inf')
    curr_t: float = starting_t
    while abs(prev_t - curr_t) > epsilon:
        iteration_count += 1
        prev_t = curr_t
        curr_t = calculate_new_threshold(channel, lower_than_hof(curr_t))

    region1: np.array = filter_channel(channel, black_mask_hoc(lower_than_hof(curr_t)))
    image1: ImageWrapper = ImageWrapper.from_dimensions(w, h, image.get_mode())
    image1.channels = [region1]

    region2: np.array = filter_channel(channel, black_mask_hoc(negation_hoc(lower_than_hof(curr_t))))
    image2: ImageWrapper = ImageWrapper.from_dimensions(w, h, image.get_mode())
    image2.channels = [region2]

    return image1, image2, int(curr_t), iteration_count
