import math

from PIL.Image import Image
import numpy as np


def apply_mask(channel: Image, x: int, y: int, mask: np.array):
    w, h = channel.size
    mask_size = mask.shape[0]
    mask_border_offset = math.floor(mask_size / 2)
    acum = 0

    for i in range(-mask_border_offset, mask_border_offset + 1):
        for j in range(-mask_border_offset, mask_border_offset + 1):
            if 0 <= x + i < w and 0 <= y + j < h:
                acum += mask[i + mask_border_offset][j + mask_border_offset] * channel.getpixel((x + i, y + j))

    return acum


def prewitt_border_detection(channel: Image):
    w, h = channel.size

    channel_cpy = channel.copy()

    prewitt_x_mask = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    prewitt_y_mask = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    for x in range(w):
        for y in range(h):
            dx = apply_mask(channel_cpy, x, y, prewitt_x_mask)
            dy = apply_mask(channel_cpy, x, y, prewitt_y_mask)
            channel.putpixel((x, y), int(math.sqrt((dx ** 2) + (dy ** 2))))

    return channel


def sobel_border_detection(channel: Image):
    w, h = channel.size

    channel_cpy = channel.copy()

    sobel_x_mask = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    sobel_y_mask = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    for x in range(w):
        for y in range(h):
            dx = apply_mask(channel_cpy, x, y, sobel_x_mask)
            dy = apply_mask(channel_cpy, x, y, sobel_y_mask)
            channel.putpixel((x, y), int(math.sqrt((dx ** 2) + (dy ** 2))))

    return channel


def laplace_border_detection(channel: Image):
    w, h = channel.size

    channel_cpy = channel.copy()

    channel_matrix = np.asarray(channel).copy()

    laplace_mask = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    for x in range(w):
        for y in range(h):
            channel_matrix[x, y] = apply_mask(channel_cpy, x, y, laplace_mask)

    for x in range(w):
        for y in range(h):
            channel.putpixel((x, y), 0)

    for x in range(w):
        for y in range(h - 1):
            if channel_matrix[x, y] == 0:
                if y - 1 >= 0 and np.sign(channel_matrix[x, y - 1]) != np.sign(channel_matrix[x, y + 1]):
                    channel.putpixel((x, y), 255)
            else:
                if np.sign(channel_matrix[x, y]) != np.sign(channel_matrix[x, y + 1]):
                    channel.putpixel((x, y), 255)

    for y in range(h):
        for x in range(w - 1):
            if channel_matrix[x, y] == 0:
                if x - 1 >= 0 and np.sign(channel_matrix[x - 1, y]) != np.sign(channel_matrix[x + 1, y]):
                    channel.putpixel((x, y), 255)
            else:
                if np.sign(channel_matrix[x, y]) != np.sign(channel_matrix[x + 1, y]):
                    channel.putpixel((x, y), 255)

    return channel


def generate_log_mask(mask_size: int, sigma: float):
    mask = np.zeros((mask_size, mask_size), dtype=float)
    mask_border_offset = math.floor(mask_size / 2)

    for x in range(-mask_border_offset, mask_border_offset + 1):
        for y in range(-mask_border_offset, mask_border_offset + 1):
            mask[x + mask_border_offset][y + mask_border_offset] = -(1 / (math.sqrt(2 * math.pi) * sigma ** 3)) * (
                        2 - (x ** 2 + y ** 2) / sigma ** 2) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return mask

