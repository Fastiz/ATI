import cv2
import numpy
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QDesktopWidget

from src.main.python import my_config
from src.main.python.components.ImageSectionSelector import ImageSectionSelector
from src.main.python.utils.ImageWrapper import ImageWrapper


class ImageSectionSelectorWithLoaderButton(QWidget):
    def __init__(self, identifier: int):
        super().__init__()

        self.handler_selection_finished = None

        self.identifier = identifier

        self.layout = QVBoxLayout()

        btnLoad = QPushButton("Load image")
        btnLoad.clicked.connect(self.onSelectButtonClicked)
        self.layout.addWidget(btnLoad)

        self.viewerLayout = QVBoxLayout()

        self.layout.addLayout(self.viewerLayout)

        self.imageSectionSelector = None
        self.image = None

        self.setLayout(self.layout)

    def subscribe_selection_finished(self, handler):
        self.handler_selection_finished = handler

    def getImage(self):
        img = self.image.pillow_image()
        selection_start, selection_end = self.imageSectionSelector.get_selection()

        if selection_start is not None:
            bigger_x, smaller_x, bigger_y, smaller_y = 0, 0, 0, 0

            if selection_start.x() >= selection_end.x():
                bigger_x = selection_start.x()
                smaller_x = selection_end.x()
            else:
                bigger_x = selection_end.x()
                smaller_x = selection_start.x()

            if selection_start.y() >= selection_end.y():
                bigger_y = selection_start.y()
                smaller_y = selection_end.y()
            else:
                bigger_y = selection_end.y()
                smaller_y = selection_start.y()

            img_left_area = (smaller_x, smaller_y, bigger_x, bigger_y)

            img_left = img.crop(img_left_area)

            return img_left
        else:
            return img

    def onSelectButtonClicked(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select image file", "",
                                                  "Images (*.jpg *.jpeg *.raw *.ppm *.pgm *.RAW)", options=options)
        if filePath:
            self.image = ImageWrapper.from_path(filePath)

            for i in reversed(range(self.viewerLayout.count())):
                self.viewerLayout.itemAt(i).widget().setParent(None)

            self.imageSectionSelector = ImageSectionSelector(self.image)

            self.viewerLayout.addWidget(self.imageSectionSelector)


class TwoImagesSelector(QWidget):
    def __init__(self, image_count: int, window_title: str = "Image Selection"):
        super().__init__()

        self.image_count = image_count

        """width, height = self.image.dimensions()
        self.setGeometry(30, 30, width, height)"""
        self.setWindowTitle(window_title)
        self.center()

        layout = QVBoxLayout()

        selectorsLayout = QHBoxLayout()

        self.selectors = [ImageSectionSelectorWithLoaderButton(i) for i in range(self.image_count)]
        for selector in self.selectors:
            selectorsLayout.addWidget(selector)

        layout.addLayout(selectorsLayout)

        self.btn_save = QPushButton("Continue")
        self.btn_save.clicked.connect(self.on_save_clicked)
        layout.addWidget(self.btn_save)

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_save_clicked(self):
        sift = cv2.SIFT_create()
        # GetImage devuelve PIL Image, no ImageWrapper

        threshold = my_config.MainWindowSelf.askForFloat("Enter distance threshold", 500)
        howManyToMatch = my_config.MainWindowSelf.askForInt("Enter minimum number of valid matches to consider equality", 50)

        open_cv_image_1 = numpy.array(self.selectors[0].getImage())[:, :, ::-1].copy()
        open_cv_image_2 = numpy.array(self.selectors[1].getImage())[:, :, ::-1].copy()

        img1 = cv2.cvtColor(open_cv_image_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(open_cv_image_2, cv2.COLOR_BGR2GRAY)

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        passing_matches = []
        for match in matches:
            if match.distance <= threshold:
                passing_matches.append(match)

        if len(passing_matches) >= howManyToMatch:
            my_config.MainWindowSelf.showMessage(f"Image matchs ({len(passing_matches)} passing matches / {len(matches)} total)", "SIFT result")
        else:
            my_config.MainWindowSelf.showMessage("Image does not match", "SIFT result")

        img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, passing_matches[:50], img2, flags=2)

        Image.fromarray(img3).show()

        """for selector in self.selectors:
            selector.getImage().show()"""
