import collections
import glob
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np

from src.main.python.utils.ImageWrapper import ImageWrapper


class BorderTracking:
    def __init__(self, start_vertex, rectangle_dim, epsilon, max_iterations=100000, theta1: Tuple[int, int, int] = None):
        self.start_vertex = (start_vertex[1], start_vertex[0])
        self.rectangle_dim = (rectangle_dim[1], rectangle_dim[0])
        self.dimensions: Tuple[int, int] = None
        self.lin: List[Tuple[int, int]] = []
        self.lout: List[Tuple[int, int]] = []
        self.channels: List[np.ndarray] = []
        self.theta1: np.array = np.array(theta1) if theta1 is not None else None
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        self.create_rectangle(start_vertex, rectangle_dim)

    def next_image(self, image: ImageWrapper) -> ImageWrapper:
        self.dimensions = image.dimensions()
        self.channels = image.channels

        if self.theta1 is None:
            self.get_avg_color()

        self.iterate()

        image_copy = image.copy()
        image_copy.channels = self.get_channels_with_borders()
        image_copy.draw_image()

        return image_copy

    def get_avg_color(self):
        w, h = self.rectangle_dim

        sampled_colors_sum = [0, 0, 0]

        sx, sy = self.start_vertex
        for x in range(sx, sx + w):
            for y in range(sy, sy + h):
                for i in range(3):
                    sampled_colors_sum[i] += self.channels[i][x, y]

        self.theta1 = np.array([float(val) / (w * h) for val in sampled_colors_sum])

    def create_rectangle(self, start_vertex, rectangle_dim):
        w, h = rectangle_dim

        for x in range(w)[1:-1]:
            self.lout.append((x, 0))
            self.lout.append((x, h-1))

        for y in range(h)[1:-1]:
            self.lout.append((0, y))
            self.lout.append((w-1, y))

        for x in range(w)[1:-1]:
            self.lin.append((x, 1))
            self.lin.append((x, h - 2))

        for y in range(h)[2:-2]:
            self.lin.append((1, y))
            self.lin.append((w - 2, y))

        sx, sy = start_vertex

        self.lin = [(x + sx, y + sy) for x, y in self.lin]
        self.lout = [(x + sx, y + sy) for x, y in self.lout]

    def iterate(self):
        max_iterations = self.max_iterations

        flag = True
        while flag and max_iterations > 0:
            flag = False
            max_iterations -= 1

            new_lout = self.lout.copy()
            new_lin = self.lin.copy()

            for x in self.lout:
                if self.fd(x) > 0:
                    flag = True
                    new_lout.remove(x)
                    new_lin.append(x)
                    for n in self.n4(x):
                        n_phi = self.phi(n, new_lin, new_lout)
                        if n_phi == 3 or n_phi == -3:
                            new_lout.append(n)

            self.lin = new_lin.copy()

            for n in self.lin:
                any_lout = False
                for n2 in self.n4(n):
                    n2_phi = self.phi(n2, new_lin, new_lout)
                    if n2_phi == -1:
                        any_lout = True
                        break

                if not any_lout:
                    new_lin.remove(n)

            self.lin = new_lin.copy()

            for x in self.lin:
                if self.fd(x) < 0:
                    flag = True
                    new_lin.remove(x)
                    new_lout.append(x)
                    for n in self.n4(x):
                        n_phi = self.phi(n, new_lin, new_lout)
                        if n_phi == -3 or n_phi == 3:
                            new_lin.append(n)

            self.lout = new_lout.copy()

            for n in self.lout:
                any_lin = False
                for n2 in self.n4(n):
                    n2_phi = self.phi(n2, new_lin, new_lout)
                    if n2_phi == 1:
                        any_lin = True
                        break

                if not any_lin:
                    new_lout.remove(n)

            self.lin = new_lin
            self.lout = new_lout

    def check_neighbors(self, n: Tuple[int, int], sign: int, lin: List[Tuple[int, int]], lout: List[Tuple[int, int]]) \
            -> bool:
        sign = int(sign / abs(sign))

        for n2 in self.n4(n):
            phi_n2 = self.phi(n2, lin, lout)
            if int(phi_n2 / abs(phi_n2)) == sign:
                return True

        return False

    def n4(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        w, h = self.dimensions

        x, y = point

        points = []

        if x - 1 >= 0:
            points.append((x - 1, y))

        if y - 1 >= 0:
            points.append((x, y - 1))

        if x + 1 < w:
            points.append((x + 1, y))

        if y + 1 < h:
            points.append((x, y + 1))

        return points

    def phi(self, x: Tuple[int, int], lin: List[Tuple[int, int]] = None, lout: List[Tuple[int, int]] = None) -> int:
        if lin is None:
            lin = self.lin
        if lout is None:
            lout = self.lout

        if x in lin:
            return 1
        elif x in lout:
            return -1
        elif self.fd(x) < 0:
            return -3
        else:
            return 3

    def fd(self, x: Tuple[int, int]) -> float:
        color: np.array = np.array([channel[x[0], x[1]] for channel in self.channels])

        theta0_norm = np.linalg.norm(self.theta1 - color)

        if theta0_norm < self.epsilon:
            return 1
        else:
            return -1

    def get_channels_with_borders(self) -> List[np.ndarray]:
        channels_copy: List[np.ndarray] = [channel.copy() for channel in self.channels]

        for x, y in self.lout:
            channels_copy[0][x, y] = 0
            channels_copy[1][x, y] = 0
            channels_copy[2][x, y] = 255

        for x, y in self.lin:
            channels_copy[0][x, y] = 255
            channels_copy[1][x, y] = 0
            channels_copy[2][x, y] = 0

        return channels_copy


def test_run():
    images = []

    folder_path = "/home/fastiz/Escritorio/video sintetico"
    for image_path in glob.glob(folder_path + "/*.jpg"):
        images.append(ImageWrapper.from_path(image_path))

    border_tracking = BorderTracking((102, 340), (200, 100), 150, theta1=(255, 0, 0))

    for image in images:
        result = border_tracking.next_image(image)
        result.draw_image()
        plt.imshow(result.image_element)
        plt.show()
