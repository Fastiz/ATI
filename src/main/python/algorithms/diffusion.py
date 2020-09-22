from typing import Callable
import numpy as np
from PIL import Image

from src.main.python.utils.ImageWrapper import ImageWrapper


def calculate_d(i: int, j: int, channel: np.array, direction: str) -> float:
    h, w = channel.shape

    if direction == 'N':
        if i+1 >= h:
            val = 0
        else:
            val = channel[i+1, j]
        return val - channel[i, j]
    elif direction == 'S':
        if i-1 < 0:
            val = 0
        else:
            val = channel[i-1, j]
        return val - channel[i, j]
    elif direction == 'E':
        if j+1 >= w:
            val = 0
        else:
            val = channel[i, j+1]
        return val - channel[i, j]
    elif direction == 'O':
        if j-1 < 0:
            val = 0
        else:
            val = channel[i, j-1]
        return val - channel[i, j]
    else:
        raise ValueError("D directions can only be N, S, E or O")


def lecrerc(x: float):
    return np.exp(-np.power(x/255, 2)/np.power(1, 2))


def calculate_direction(i: int, j: int, channel: np.array, direction: str,
                        c_calculator: Callable[[float], float]):
    val = calculate_d(i, j, channel, direction)
    return val * c_calculator(val)


def diffusion_step(c_calculator: Callable[[float], float], image: ImageWrapper, step: float) -> ImageWrapper:
    w, h = image.dimensions()
    new_image: ImageWrapper = ImageWrapper.from_dimensions(w, h, image.get_mode())

    new_channels = [np.zeros(shape=(h, w))] * len(image.channels)
    for (channel, new_channel) in zip(image.channels, new_channels):
        for i in range(h):
            for j in range(w):
                new_channel[i, j] = channel[i, j] + step * (
                    calculate_direction(i, j, channel, 'N', c_calculator) +
                    calculate_direction(i, j, channel, 'S', c_calculator) +
                    calculate_direction(i, j, channel, 'E', c_calculator) +
                    calculate_direction(i, j, channel, 'O', c_calculator)
                )

    new_image.channels = new_channels
    new_image.draw_image()
    return new_image


def anisotropic_diffusion_step(image: Image, step=1.0/4) -> ImageWrapper:
    return diffusion_step(lecrerc, image, step)


def isotropic_diffusion_step(image: Image, step=1.0/4) -> ImageWrapper:
    return diffusion_step(lambda d_val: 1, image, step)
