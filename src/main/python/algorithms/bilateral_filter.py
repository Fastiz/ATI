from typing import Tuple, List
import numpy as np
from src.main.python.utils.ImageWrapper import ImageWrapper


def calculate_w(i: int, j: int, w: int, k: int, sigma_s: float, sigma_r: float, channels: List[np.array]) -> float:
    ij_value = np.array([channel[i, j]/255 for channel in channels])
    wk_value = np.array([channel[w, k]/255 for channel in channels])

    val_1: float = -(np.power(i - k, 2) + np.power(j - w, 2)) / (2 * np.power(sigma_s, 2))
    val_2: float = -(np.power(np.linalg.norm(ij_value - wk_value), 2)) / (2 * np.power(sigma_r, 2))
    return np.exp(val_1 + val_2)


def calculate_filter(channels: List[np.array], i: int, j: int, window_dim: Tuple[int, int],
                     sigma_s: float, sigma_r: float) -> List[float]:
    window_w, window_h = window_dim
    image_w, image_h = channels[0].shape

    max_h = int((window_h - 1)/2)
    max_w = int((window_w - 1)/2)

    nominators: List[float] = [0] * len(channels)
    denominators: List[float] = [0] * len(channels)
    for k in range(max(0, i - max_h), min(image_h, i + max_h + 1)):
        for w in range(max(0, j - max_w), min(image_w, j + max_w + 1)):
            aux = calculate_w(i, j, w, k, sigma_s, sigma_r, channels)
            for i, channel in zip(range(len(channels)), channels):
                nominators[i] += channel[k, w] * aux
                denominators[i] += aux

    return [nominator/denominator for nominator, denominator in zip(nominators, denominators)]


def bilateral_filter(image: ImageWrapper, window_dim: Tuple[int, int], sigma_s: float, sigma_r: float) -> ImageWrapper:
    window_w, window_h = window_dim
    if window_w % 2 == 0 or window_h % 2 == 0:
        raise ValueError("Window dimensions should be odd numbers")

    w, h = image.dimensions()
    new_image: ImageWrapper = ImageWrapper.from_dimensions(w, h, image.get_mode())

    new_channels = [np.zeros(shape=(h, w))] * len(image.channels)
    for i in range(h):
        for j in range(w):
            values = calculate_filter(image.channels, i, j, window_dim, sigma_s, sigma_r)
            for val, channel in zip(values, new_channels):
                channel[i, j] = val

    new_image.channels = new_channels
    new_image.draw_image()
    return new_image
