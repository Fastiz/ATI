import os

import PIL
from PIL import Image


def is_raw(fileextension):
    return fileextension == ".pgm" or fileextension == ".raw" or fileextension == ".RAW"


def load_raw(file_path):
    image_data = open(file_path, "rb").read()
    return Image.frombytes('L', (290, 207), image_data)


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
    def from_dimensions(w, h):
        return ImageWrapper(Image.new('RGB', (w, h)))

    def get_pixel(self, x, y):
        return self.image_element.getpixel((x, y))

    def set_pixel(self, x, y, value):
        return self.image_element.putpixel((x, y), value)

    def dimensions(self):
        return self.image_element.size

    def file_name(self):
        return self.filename

    def file_path(self):
        return self.file_path

    def pillow_image(self):
        return self.image_element

    def __is_raw(self):
        return is_raw(self.fileextension)

    def __load_raw(self):
        return load_raw(self.file_path)
