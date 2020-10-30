from typing import Tuple, List

from PIL.Image import Image

from src.main.python.algorithms.border_detection import apply_mask
from src.main.python.algorithms.channel_operations import channel_gaussian_window
from src.main.python.utils import ImageWrapper
import numpy as np

import matplotlib.pyplot as plt


def directional_derivatives(channel: np.ndarray) -> Tuple[np.array, np.array]:
    sobel_x_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

    sobel_y_mask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    gx = np.ndarray(shape=channel.shape)
    gy = np.ndarray(shape=channel.shape)

    w, h = channel.shape
    for x in range(w):
        for y in range(h):
            gx[x, y] = apply_mask(channel, y, x, sobel_x_mask)
            gy[x, y] = apply_mask(channel, y, x, sobel_y_mask)

    return gx, gy


def in_range(w, h, pos: np.array):
    x, y = pos
    if 0 <= x < w and 0 <= y < h:
        return True
    return False


def maximum_suppression(g: np.ndarray, slope: np.ndarray):
    result = g.copy()

    w, h = g.shape
    for x in range(w):
        for y in range(h):
            pos = np.array([x, y])

            side1 = np.add(pos, slope[x, y])
            if in_range(w, h, side1):
                if g[side1[0], side1[1]] > g[x, y]:
                    result[x, y] = 0

            side2 = np.subtract(pos, slope[x, y])
            if in_range(w, h, side2):
                if g[side2[0], side2[1]] > g[x, y]:
                    result[x, y] = 0

    return result


def discrete_slope(slope: np.ndarray):
    result = np.ndarray(shape=slope.shape, dtype=object)

    w, h = slope.shape
    for x in range(w):
        for y in range(h):
            s = slope[x, y]
            if s < 0:
                s += 180
            
            if (0 <= s < 22.5) or (157.5 <= s <= 180):
                result[x, y] = np.array([1, 0])
            elif 22.5 <= s < 67.5:
                result[x, y] = np.array([-1, 1])
            elif 67.5 <= s < 112.5:
                result[x, y] = np.array([0, 1])
            elif 112.5 <= s < 157.5:
                result[x, y] = np.array([-1, -1])
            else:
                raise Exception("out of range")

    return result


def mark_borders(t: int, channel: np.ndarray):
    borders = np.ndarray(shape=channel.shape, dtype=bool)

    w, h = channel.shape
    for x in range(w):
        for y in range(h):
            if channel[x, y] >= t:
                borders[x, y] = True
            else:
                borders[x, y] = False

    return borders


def is_connected(borders, x, y):
    w, h = borders.shape

    for j in range(x - 1, x + 2):
        for i in range(y - 1, y + 2):
            if i == y and j == x:
                continue

            if in_range(w, h, (j, i)) and borders[j, i]:
                return True

    return False


def weak_borders(t: int, borders: np.ndarray, channel: np.ndarray):
    modified = False

    w, h = channel.shape
    for x in range(w):
        for y in range(h):
            if (not borders[x, y]) and channel[x, y] >= t and is_connected(borders, x, y):
                borders[x, y] = True
                modified = True

    if modified:
        return weak_borders(t, borders, channel)

    return borders


def clean_image_with_borders(borders: np.ndarray, channel: np.ndarray):
    w, h = channel.shape
    for x in range(w):
        for y in range(h):
            if borders[x, y]:
                channel[x, y] = 255
            else:
                channel[x, y] = 0

    return channel


def canny_border_detection(image: ImageWrapper.ImageWrapper, gauss_sigma, t1, t2) -> ImageWrapper.ImageWrapper:
    img: Image = image.image_element

    # Applying gaussian filter
    img = channel_gaussian_window(img, gauss_sigma)

    # Magnitude and slope
    gx, gy = directional_derivatives(ImageWrapper.ImageWrapper(img).channels[0])

    g = np.hypot(gx, gy)

    g_max = g.max()
    g_min = g.min()

    g_w, g_h = g.shape

    for x in range(g_w):
        for y in range(g_h):
            g[x, y] = (g[x, y] - g_min) * 255 / (g_max - g_min)

    slope = discrete_slope(np.array([val * 180 / np.pi for val in np.arctan2(gy, gx)]))

    channel = maximum_suppression(g, slope)

    borders = mark_borders(t2, channel)

    borders = weak_borders(t1, borders, channel)

    channel = clean_image_with_borders(borders, channel)

    image_copy = image.copy()
    image_copy.channels = [channel]

    return image_copy
