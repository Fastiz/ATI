import io
import os

import PIL
from PIL import Image
from PyQt5.QtWidgets import QMessageBox


class ImageWrapper:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.filename = os.path.basename(self.file_path)
        self.fileextension = os.path.splitext(file_path)[1]

        if not self.__is_raw():
            self.image_element = PIL.Image.open(file_path)
        else:
            self.image_element = self.__load_raw()

    def dimensions(self):
        return self.image_element.size

    def file_name(self):
        return self.filename

    def file_path(self):
        return self.file_path

    def pillow_image(self):
        return self.image_element

    def __is_raw(self):
        return self.fileextension == ".pgm" or self.fileextension == ".raw"

    def __load_raw(self):
        image_data = open(self.file_path, "rb").read()
        return Image.frombytes('L', (512, 512), image_data)
