import math
from typing import List

from src.main.python.algorithms.canny_border_detection import directional_derivatives
from src.main.python.utils.ImageWrapper import ImageWrapper
import numpy as np
import matplotlib.pyplot as plt


def sliding_window(channel: np.ndarray, window):
    h, w = channel.shape

    window_size = len(window)
    window_border_offset = math.floor(window_size / 2)

    for x in range(w):
        for y in range(h):
            px_val = 0.0
            for i in range(-window_border_offset, window_border_offset+1):
                for j in range(-window_border_offset, window_border_offset+1):
                    if 0 <= x + i < w and 0 <= y + j < h:
                        px_val += window[i + window_border_offset][j + window_border_offset] * channel[y+j, x+i]
            channel[y, x] = int(px_val)

    return channel


def gaussian_window(channel: np.ndarray, sigma: float):
    window_size = int(2 * sigma + 1)
    window_border_offset = math.floor(window_size / 2)
    window = np.full((window_size, window_size), 0, dtype=float)

    factor = 1/(2*math.pi*sigma*sigma)
    exp_factor = -1/(sigma*sigma)
    for i in range(-window_border_offset, window_border_offset + 1):
        for j in range(-window_border_offset, window_border_offset + 1):
            window[i+window_border_offset][j+window_border_offset] = factor * math.exp(exp_factor * ((i**2) + (j**2)))

    sum = window.sum()
    for row in window:
        for i in range(len(row)):
            row[i] /= sum

    return sliding_window(channel, window)


def harris_corner_detection(img: ImageWrapper) -> ImageWrapper:
    channel = img.channels[0]

    gx, gy = directional_derivatives(channel)

    ix2 = np.ndarray(shape=channel.shape)
    iy2 = np.ndarray(shape=channel.shape)
    ixy = np.ndarray(shape=channel.shape)

    h, w = channel.shape

    for x in range(w):
        for y in range(h):
            ix2[x, y] = np.power(gx[x, y], 2)
            iy2[x, y] = np.power(gy[x, y], 2)
            ixy[x, y] = gx[x, y] * gy[x, y]

    ix2 = gaussian_window(ix2, 1)
    iy2 = gaussian_window(iy2, 1)
    ixy = gaussian_window(ixy, 1)

    k = 0.04

    r = np.ndarray(shape=channel.shape)

    for x in range(w):
        for y in range(h):
            r[x, y] = (ix2[x, y] * iy2[x, y] - np.power(ixy[x, y], 2)) - k * np.power(ix2[x, y] + iy2[x, y], 2)

    r_max = r.max()

    print(r_max)

    new_img = ImageWrapper.from_dimensions(w, h, 'RGB')
    for x in range(w):
        for y in range(h):
            if r[x, y] >= 0.002 * r_max:
                new_img.channels[0][x, y] = 255
                new_img.channels[1][x, y] = 0
                new_img.channels[2][x, y] = 0
            else:
                for i in range(3):
                    new_img.channels[i][x, y] = channel[x, y]

    new_img.draw_image()

    return new_img


def harris_test():
    img = ImageWrapper.from_path("/home/fastiz/Documentos/Facultad/ATI/images/TEST.pgm")
    plt.imshow(harris_corner_detection(img).image_element)

    # img.channels = [gaussian_window(img.channels[0], 2)]
    # img.draw_image()
    #
    # plt.imshow(img.image_element)
    plt.show()


harris_test()