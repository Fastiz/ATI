import math
from typing import List

from PIL.Image import Image
import numpy as np


def apply_mask(channel: Image, x: int, y: int, mask: np.array):
    h, w = channel.shape
    mask_size = mask.shape[0]
    mask_border_offset = math.floor(mask_size / 2)
    acum = 0

    for i in range(-mask_border_offset, mask_border_offset + 1):
        for j in range(-mask_border_offset, mask_border_offset + 1):
            if 0 <= x + i < w and 0 <= y + j < h:
                acum += mask[i + mask_border_offset][j + mask_border_offset] * channel[y + j, x + i]

    return acum


def first_derivative_border_detection(channel: np.ndarray, derivative_masks: List[np.ndarray]):
    h, w = channel.shape

    channel_cpy = channel.copy()

    for x in range(w):
        for y in range(h):
            derivative_result = 0
            for mask in derivative_masks:
                derivative_result += apply_mask(channel_cpy, x, y, mask) ** 2
            channel[y, x] = math.sqrt(derivative_result)
            if channel[y, x] > 255:
                channel[y, x] = 255

    return channel


def laplace_border_detection(channel: np.ndarray, mask: np.ndarray, threshold: float = 0):
    h, w = channel.shape

    channel_cpy = channel.copy()

    channel_matrix = channel.copy()

    for x in range(w):
        for y in range(h):
            channel_matrix[y, x] = apply_mask(channel_cpy, x, y, mask)

    for x in range(w):
        for y in range(h):
            channel[y, x] = 0

    # if abs(channel_matrix[y, x]) + abs(channel_matrix[x + 1, y]) >= threshold:

    for x in range(w):
        for y in range(h - 1):
            if channel_matrix[y, x] == 0:
                if y - 1 >= 0 and np.sign(channel_matrix[y - 1, x]) != np.sign(channel_matrix[y + 1, x]):
                    if abs(channel_matrix[y - 1, x]) + abs(channel_matrix[y + 1, x]) >= threshold:
                        channel[y, x] = 255
            else:
                if np.sign(channel_matrix[y, x]) != np.sign(channel_matrix[y + 1, x]):
                    if abs(channel_matrix[y, x]) + abs(channel_matrix[y + 1, x]) >= threshold:
                        channel[y, x] = 255

    for y in range(h):
        for x in range(w - 1):
            if channel_matrix[y, x] == 0:
                if x - 1 >= 0 and np.sign(channel_matrix[y, x - 1]) != np.sign(channel_matrix[y, x + 1]):
                    if abs(channel_matrix[y, x - 1]) + abs(channel_matrix[y, x + 1]) >= threshold:
                        channel[y, x] = 255
            else:
                if np.sign(channel_matrix[y, x]) != np.sign(channel_matrix[y, x + 1]):
                    if abs(channel_matrix[y, x]) + abs(channel_matrix[y, x + 1]) >= threshold:
                        channel[y, x] = 255

    return channel


def generate_log_mask(mask_size: int, sigma: float):
    mask = np.zeros((mask_size, mask_size), dtype=float)
    mask_border_offset = math.floor(mask_size / 2)

    for x in range(-mask_border_offset, mask_border_offset + 1):
        for y in range(-mask_border_offset, mask_border_offset + 1):
            mask[x + mask_border_offset][y + mask_border_offset] = -(1 / (math.sqrt(2 * math.pi) * sigma ** 3)) * (
                    2 - (x ** 2 + y ** 2) / sigma ** 2) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return mask


def rotate_matrix(mat):
    if not len(mat):
        return

    top = 0
    bottom = len(mat) - 1

    left = 0
    right = len(mat[0]) - 1

    while left < right and top < bottom:
        prev = mat[top + 1][left]

        for i in range(left, right + 1):
            curr = mat[top][i]
            mat[top][i] = prev
            prev = curr

        top += 1

        for i in range(top, bottom + 1):
            curr = mat[i][right]
            mat[i][right] = prev
            prev = curr

        right -= 1

        for i in range(right, left - 1, -1):
            curr = mat[bottom][i]
            mat[bottom][i] = prev
            prev = curr

        bottom -= 1

        for i in range(bottom, top - 1, -1):
            curr = mat[i][left]
            mat[i][left] = prev
            prev = curr

        left += 1

    return mat
