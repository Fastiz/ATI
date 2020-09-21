from PIL.Image import Image
import math
import matplotlib.pyplot as plt
import numpy as np

import src.main.python.algorithms.border_detection as bd

def channel_mean(channel: Image):
    h, w = channel.size

    sum = 0

    for x in range(w):
        for y in range(h):
            sum += channel.getpixel((y, x))

    return sum / (h * w)


def channel_variance(channel: Image):
    h, w = channel.size
    mean = channel_mean(channel)

    sum = 0
    for x in range(w):
        for y in range(h):
            sum += (channel.getpixel((y, x)) - mean) ** 2

    return sum / (h * w)


def channel_standard_deviation(channel: Image):
    return math.sqrt(channel_variance(channel))


def channel_contrast(channel: Image, k: int):
    h, w = channel.size
    mean = channel_mean(channel)
    deviation = channel_standard_deviation(channel)

    r1 = mean - deviation
    r2 = mean + deviation

    s1 = r1 / k
    s2 = 255 - (255 - r2) / k

    for x in range(w):
        for y in range(h):
            p = channel.getpixel((y, x))
            if p < r1:
                (x1, x2), (y1, y2) = (0, r1), (0, s1)
            elif r1 <= p <= r2:
                (x1, x2), (y1, y2) = (r1, r2), (s1, s2)
            else:
                (x1, x2), (y1, y2) = (r2, 255), (s2, 255)

            new_p = round((y2 - y1) * (p - x1) / (x2 - x1) + y1)
            channel.putpixel((y, x), new_p)

    return channel


def channel_threshold(channel: Image, u: int):  # Umbralización
    h, w = channel.size
    for x in range(w):
        for y in range(h):
            p = channel.getpixel((y, x))
            if p < u:
                channel.putpixel((y, x), 0)
            else:
                channel.putpixel((y, x), 255)

    return channel


def channel_negative(channel: Image):
    h, w = channel.size
    for x in range(w):
        for y in range(h):
            p = channel.getpixel((y, x))
            channel.putpixel((y, x), 255 - p)

    return channel


def graph_histogram(channel: Image):
    data = []
    h, w = channel.size
    for x in range(w):
        for y in range(h):
            data.append(channel.getpixel((y, x)))

    plt.hist(data, bins=256, density=True)
    plt.show()


def channel_histogram(channel: Image, display: bool = False):
    data = [0] * 256

    h, w = channel.size
    for x in range(w):
        for y in range(h):
            data[channel.getpixel((y, x))] += 1

    for i in range(len(data)):
        data[i] /= h * w

    if display:
        graph_histogram(channel)

    return data


def channel_sliding_window(channel: Image, window):
    h, w = channel.size

    window_size = len(window)
    window_border_offset = math.floor(window_size / 2)

    for x in range(w):
        for y in range(h):
            px_val = 0.0
            for i in range(-window_border_offset, window_border_offset+1):
                for j in range(-window_border_offset, window_border_offset+1):
                    if 0 <= x + i < w and 0 <= y + j < h:
                        px_val += window[i + window_border_offset][j + window_border_offset] * channel.getpixel((y+j, x+i))
            channel.putpixel((y, x), int(px_val))

    return channel

def channel_mean_window(channel: Image, window_size: int):
    window = np.full((window_size, window_size), 1/(window_size*window_size), dtype=float)
    return channel_sliding_window(channel, window)


def channel_gaussian_window(channel: Image, sigma: float):
    h, w = channel.size

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

    return channel_sliding_window(channel, window)


def channel_median_window(channel: Image, window_size: int):
    h, w = channel.size

    window_border_offset = math.floor(window_size / 2)

    for x in range(w):
        for y in range(h):
            px_val_list = []
            for i in range(-window_border_offset, window_border_offset+1):
                for j in range(-window_border_offset, window_border_offset+1):
                    px_val_list.append(channel.getpixel((y+j, x+i)) if 0 <= x + i < w and 0 <= y + j < h else 0)
            channel.putpixel((y, x), int(np.median(px_val_list)))

    return channel


def channel_ponderated_median_window_3x3(channel: Image):
    h, w = channel.size

    window_size = 3
    window_border_offset = math.floor(window_size / 2)

    window = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]

    for x in range(w):
        for y in range(h):
            px_val_list = []
            for i in range(-window_border_offset, window_border_offset+1):
                for j in range(-window_border_offset, window_border_offset+1):
                    for k in range(window[i + window_border_offset][j + window_border_offset]):
                        px_val_list.append(channel.getpixel((y+j, x+i)) if 0 <= x + i < w and 0 <= y + j < h else 0)
            channel.putpixel((y, x), int(np.median(px_val_list)))

    return channel


def channel_highpass_window(channel: Image, window_size: int):
    window = np.full((window_size, window_size), -1, dtype=float)
    center = math.floor(window_size/2)

    window[center][center] = (window_size**2) - 1

    return channel_sliding_window(channel, window)



