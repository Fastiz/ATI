from typing import Tuple
import numpy as np
from enum import Enum

from src.main.python.utils.ImageWrapper import ImageWrapper


def circular_mask(radius: int) -> np.array:
    result = []

    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            distance = np.sqrt(x ** 2 + y ** 2)
            if distance <= radius:
                result.append((x, y))

    return np.array(result)


def c(r: Tuple[int, int], r0: Tuple[int, int], t: int, channel: np.ndarray) -> int:
    rx, ry = r
    r0x, r0y = r0

    i = channel[rx, ry]
    i0 = channel[r0x, r0y]

    return 1 if abs(i - i0) < t else 0


def in_range(w, h, pos: np.array):
    x, y = pos
    if 0 <= x < w and 0 <= y < h:
        return True
    return False


def s(r0: Tuple[int, int], t: int, channel: np.ndarray, mask: np.array) -> float:
    w, h = channel.shape

    rx, ry = r0

    moved_mask = [(rx + x, ry + y) for x, y in mask]

    moved_mask = filter(lambda pos: in_range(w, h, pos), moved_mask)

    n = sum([c(r, r0, t, channel) for r in moved_mask])

    return 1.0 - float(n) / len(mask)


class Detection(Enum):
    corner = 1
    border = 2
    none = 3


def susan_border_detection(t: int, channel: np.ndarray, s_border: float, s_corner: float) -> np.ndarray:
    w, h = channel.shape

    result = np.ndarray(shape=channel.shape, dtype=object)

    mask = circular_mask(3)

    for x in range(w):
        for y in range(h):
            aux = s((x, y), t, channel, mask)

            if aux > s_corner:
                result[x, y] = Detection.corner
            elif aux > s_border:
                result[x, y] = Detection.border
            else:
                result[x, y] = Detection.none

    return result


def apply_susan_border_detection(t: int, image: ImageWrapper, s_border: float, s_corner: float) -> ImageWrapper:
    channel = image.channels[0]

    channels = [channel.copy(), channel.copy(), channel.copy()]

    borders = susan_border_detection(t, channel, s_border, s_corner)

    w, h = channel.shape

    for x in range(w):
        for y in range(h):
            if borders[x, y] == Detection.corner:
                channels[0][x, y] = 0
                channels[1][x, y] = 255
                channels[2][x, y] = 0
            elif borders[x, y] == Detection.border:
                channels[0][x, y] = 0
                channels[1][x, y] = 0
                channels[2][x, y] = 255
            else:
                channels[0][x, y] = channel[x, y]
                channels[1][x, y] = channel[x, y]
                channels[2][x, y] = channel[x, y]

    image.channels = channels
    image.mode = 'RGB'

    return image
