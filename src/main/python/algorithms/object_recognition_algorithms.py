from typing import List

import cv2
import numpy
from PIL import Image

from src.main.python import my_config
from src.main.python.components.TwoImageSelector import ImageSectionSelectorWithLoaderButton


def kaze(selectors: List[ImageSectionSelectorWithLoaderButton]):
    method = cv2.KAZE_create()
    # GetImage devuelve PIL Image, no ImageWrapper

    threshold = my_config.MainWindowSelf.askForFloat("Enter distance threshold", 500)
    howManyToMatch = my_config.MainWindowSelf.askForInt("Enter minimum number of valid matches to consider equality",
                                                        50)

    open_cv_image_1 = numpy.array(selectors[0].getImage())[:, :, ::-1].copy()
    open_cv_image_2 = numpy.array(selectors[1].getImage())[:, :, ::-1].copy()

    img1 = cv2.cvtColor(open_cv_image_1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(open_cv_image_2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = method.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = method.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    passing_matches = []
    for match in matches:
        if match.distance <= threshold:
            passing_matches.append(match)

    if len(passing_matches) >= howManyToMatch:
        my_config.MainWindowSelf.showMessage(
            f"Image matchs ({len(passing_matches)} passing matches / {len(matches)} total)", "SIFT result")
    else:
        my_config.MainWindowSelf.showMessage("Image does not match", "SIFT result")

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, passing_matches[:50], img2, flags=2)

    Image.fromarray(img3).show()


def akaze(selectors: List[ImageSectionSelectorWithLoaderButton]):
    method = cv2.AKAZE_create()
    # GetImage devuelve PIL Image, no ImageWrapper

    threshold = my_config.MainWindowSelf.askForFloat("Enter distance threshold", 500)
    howManyToMatch = my_config.MainWindowSelf.askForInt("Enter minimum number of valid matches to consider equality",
                                                        50)

    open_cv_image_1 = numpy.array(selectors[0].getImage())[:, :, ::-1].copy()
    open_cv_image_2 = numpy.array(selectors[1].getImage())[:, :, ::-1].copy()

    img1 = cv2.cvtColor(open_cv_image_1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(open_cv_image_2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = method.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = method.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    passing_matches = []
    for match in matches:
        if match.distance <= threshold:
            passing_matches.append(match)

    if len(passing_matches) >= howManyToMatch:
        my_config.MainWindowSelf.showMessage(
            f"Image matchs ({len(passing_matches)} passing matches / {len(matches)} total)", "SIFT result")
    else:
        my_config.MainWindowSelf.showMessage("Image does not match", "SIFT result")

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, passing_matches[:50], img2, flags=2)

    Image.fromarray(img3).show()


def sift(selectors: List[ImageSectionSelectorWithLoaderButton]):
    method = cv2.xfeatures2d.SIFT_create()
    # GetImage devuelve PIL Image, no ImageWrapper

    threshold = my_config.MainWindowSelf.askForFloat("Enter distance threshold", 500)
    howManyToMatch = my_config.MainWindowSelf.askForInt("Enter minimum number of valid matches to consider equality",
                                                        50)

    open_cv_image_1 = numpy.array(selectors[0].getImage())[:, :, ::-1].copy()
    open_cv_image_2 = numpy.array(selectors[1].getImage())[:, :, ::-1].copy()

    img1 = cv2.cvtColor(open_cv_image_1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(open_cv_image_2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = method.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = method.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    passing_matches = []
    for match in matches:
        if match.distance <= threshold:
            passing_matches.append(match)

    if len(passing_matches) >= howManyToMatch:
        my_config.MainWindowSelf.showMessage(
            f"Image matchs ({len(passing_matches)} passing matches / {len(matches)} total)", "SIFT result")
    else:
        my_config.MainWindowSelf.showMessage("Image does not match", "SIFT result")

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, passing_matches[:50], img2, flags=2)

    Image.fromarray(img3).show()


def surf(selectors: List[ImageSectionSelectorWithLoaderButton]):
    method = cv2.xfeatures2d.SURF_create()
    # GetImage devuelve PIL Image, no ImageWrapper

    threshold = my_config.MainWindowSelf.askForFloat("Enter distance threshold", 500)
    howManyToMatch = my_config.MainWindowSelf.askForInt(
        "Enter minimum number of valid matches to consider equality",
        50)

    open_cv_image_1 = numpy.array(selectors[0].getImage())[:, :, ::-1].copy()
    open_cv_image_2 = numpy.array(selectors[1].getImage())[:, :, ::-1].copy()

    img1 = cv2.cvtColor(open_cv_image_1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(open_cv_image_2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = method.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = method.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    passing_matches = []
    for match in matches:
        if match.distance <= threshold:
            passing_matches.append(match)

    if len(passing_matches) >= howManyToMatch:
        my_config.MainWindowSelf.showMessage(
            f"Image matchs ({len(passing_matches)} passing matches / {len(matches)} total)", "SIFT result")
    else:
        my_config.MainWindowSelf.showMessage("Image does not match", "SIFT result")

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, passing_matches[:50], img2, flags=2)

    Image.fromarray(img3).show()
