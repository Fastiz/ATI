import os

import PIL
from PIL import Image

from src.main.python import my_config


def is_raw(fileextension):
    return fileextension == ".raw" or fileextension == ".RAW"


def load_raw(file_path):
    image_data = open(file_path, "rb").read()

    w = my_config.MainWindowSelf.askForInt("Enter image width", 256)
    h = my_config.MainWindowSelf.askForInt("Enter image height", 256)

    return Image.frombytes('L', (w, h), image_data)


class ImageWrapper:
    def __init__(self, image: PIL.Image.Image, file_path=None, filename=None, fileextension=None):
        self.image_element = image
        self.file_path = file_path
        self.filename = filename
        self.fileextension = fileextension

    @staticmethod
    def from_path(file_path):
        file_path = file_path

        filename = os.path.basename(file_path)
        fileextension = os.path.splitext(file_path)[1]

        if not is_raw(fileextension):
            image_element = PIL.Image.open(file_path)
        else:
            image_element = load_raw(file_path)

        return ImageWrapper(image_element, file_path, filename, fileextension)

    @staticmethod
    def from_dimensions(w, h, mode='RGB'):
        mode = 'RGB' if mode is None else mode
        return ImageWrapper(Image.new(mode, (w, h)))

    def copy(self):
        return ImageWrapper(self.image_element.copy(), self.file_path, self.filename, self.fileextension)

    def get_pixel(self, x, y):
        val = self.image_element.getpixel((x, y))

        if isinstance(val, int):
            return [val]
        return val

    def set_pixel(self, x, y, value):
        return self.image_element.putpixel((x, y), value if len(value) > 1 else value[0])

    def dimensions(self):
        return self.image_element.size

    def file_name(self):
        return self.filename

    def file_path(self):
        return self.file_path

    def pillow_image(self):
        return self.image_element

    def set_pillow_image(self, image: Image):
        self.image_element = image

    def __is_raw(self):
        return is_raw(self.fileextension)

    def __load_raw(self):
        return load_raw(self.file_path)

    def get_image_element(self):
        return self.image_element

    def get_mode(self):
        return self.image_element.mode
