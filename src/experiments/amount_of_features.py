import matplotlib.pyplot as plt
import cv2

from src.experiments.utils import img_from_file


def run():
    imgs = [img_from_file('../../dataset/img' + str(i) + '.ppm') for i in range(1, 7)]

    fig = plt.figure()

    labels = ['Sift', 'Surf', 'Kaze', 'Akaze']
    values = []

    for method in [cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SURF_create(), cv2.KAZE_create(), cv2.AKAZE_create()]:
        sum_value = 0
        for img in imgs:
            keypoints, descriptors = method.detectAndCompute(img, None)
            sum_value += len(descriptors)
        values.append(sum_value / float(len(imgs)))

    plt.bar(labels, values)

run()
plt.show()

