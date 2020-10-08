from typing import Callable
import numpy as np
from PIL import Image

from src.main.python.utils.ImageWrapper import ImageWrapper


def calculate_d(x: int, y: int, channel: np.array, direction: str) -> float:
    w, h = channel.shape

    if direction == 'N':
        if x+1 >= h:
            val = 0
        else:
            val = channel[x+1, y]
        return val - channel[x, y]
    elif direction == 'S':
        if x-1 < 0:
            val = 0
        else:
            val = channel[x-1, y]
        return val - channel[x, y]
    elif direction == 'E':
        if y+1 >= w:
            val = 0
        else:
            val = channel[x, y+1]
        return val - channel[x, y]
    elif direction == 'O':
        if y-1 < 0:
            val = 0
        else:
            val = channel[x, y-1]
        return val - channel[x, y]
    else:
        raise ValueError("D directions can only be N, S, E or O")


def lecrerc_hof(sigma: float) -> Callable[[float], float]:
    def lecrerc(x: float):
        return np.exp(-np.power(x, 2)/np.power(sigma, 2))
    return lecrerc


def calculate_direction(x: int, y: int, channel: np.array, direction: str,
                        c_calculator: Callable[[float], float]):
    val = calculate_d(x, y, channel, direction)
    return val * c_calculator(val)


def diffusion_step(c_calculator: Callable[[float], float], image: ImageWrapper, step: float) -> ImageWrapper:
    w, h = image.dimensions()
    new_image: ImageWrapper = ImageWrapper.from_dimensions(w, h, image.get_mode())
    
    new_channels = []
    for channel in image.channels:
        new_channel = np.zeros(shape=(w, h))
        for x in range(w):
            for y in range(h):
                new_channel[x, y] = channel[x, y] + step * (
                    calculate_direction(x, y, channel, 'N', c_calculator) +
                    calculate_direction(x, y, channel, 'S', c_calculator) +
                    calculate_direction(x, y, channel, 'E', c_calculator) +
                    calculate_direction(x, y, channel, 'O', c_calculator)
                )
        new_channels.append(new_channel)

    new_image.channels = new_channels
    new_image.draw_image()
    return new_image


def anisotropic_diffusion_step(image: Image, sigma: float, step=1.0/4) -> ImageWrapper:
    return diffusion_step(lecrerc_hof(sigma), image, step)


def isotropic_diffusion_step(image: Image, step=1.0/4) -> ImageWrapper:
    return diffusion_step(lambda d_val: 1, image, step)
