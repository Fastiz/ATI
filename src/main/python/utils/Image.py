import re
import rawpy
import numpy as np
from PIL import Image


def load_image_array(file_path):
    if re.match(".(raw|RAW)$", file_path):
        return load_raw_to_array(file_path)
    else:
        im = Image.open(file_path)
        result = np.asarray(im)
        return result


def load_raw_to_array(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()

    return np.asarray(rgb)


def save_image_array(file_path, img):
    im = Image.fromarray(img)
    im.save(file_path + "test.jpg")
