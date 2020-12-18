import matplotlib.pyplot as plt
import cv2

from src.experiments.utils import img_from_file


def run():
    img = img_from_file('../../images/Lenaclor.ppm')

    fig = plt.figure()

    labels = ['Sift', 'Surf', 'Kaze', 'Akaze']
    values = []

    for method in [cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SURF_create(), cv2.KAZE_create(), cv2.AKAZE_create()]:
        keypoints, descriptors = method.detectAndCompute(img, None)
        values.append(len(descriptors))

    plt.bar(labels, values)

