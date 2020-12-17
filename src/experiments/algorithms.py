import cv2


def base_algorithm(method, img1, img2):
    keypoints_1, descriptors_1 = method.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = method.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def sift(img1, img2):
    return base_algorithm(cv2.xfeatures2d.SIFT_create(), img1, img2)


def surf(img1, img2):
    return base_algorithm(cv2.xfeatures2d.SURF_create(), img1, img2)


def kaze(img1, img2):
    return base_algorithm(cv2.KAZE_create(), img1, img2)


def akaze(img1, img2):
    return base_algorithm(cv2.AKAZE_create(), img1, img2)


def img_from_file(path):
    return cv2.imread(path)
