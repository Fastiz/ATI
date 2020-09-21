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
                acum += mask[i + mask_border_offset][j + mask_border_offset] * channel.getpixel((x+i, y + j))

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


def asd(channel: Image):
    w, h = channel.size

    matrix = np.asarray(channel)

    gx = np.array(([-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]))
    gy = np.array(([-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]))
    for y in range(2, h - 2):

        for x in range(2, w - 2):
            r_gx, r_gy = 0, 0
            for kernel_offset_y in range(-1, 1 + 1):

                for kernel_offset_x in range(-1, 1 + 1):

                    xx = x + kernel_offset_x
                    yy = y + kernel_offset_y
                    color = matrix[xx, yy]
                    # print(kernel_offset_y, kernel_offset_x)
                    if kernel_offset_x != 0:
                        k = gx[kernel_offset_x + 1,
                                    kernel_offset_y + 1]
                        r_gx += color[0] * k

                    if kernel_offset_y != 0:
                        k = gy[kernel_offset_x + 1,
                                    kernel_offset_y + 1]
                        r_gy += color[0] * k

            magnitude = math.sqrt(r_gx ** 2 + r_gy ** 2)
            # update the pixel if the magnitude is above threshold else black pixel
            matrix[x, y] = magnitude if magnitude > 0 else 0
        # cap the values
    np.putmask(matrix, matrix > 255, 255)
    np.putmask(matrix, matrix < 0, 0)