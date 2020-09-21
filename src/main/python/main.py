import os
import sys

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QPushButton, QInputDialog, QTabWidget, QGridLayout

from src.main.python import my_config
from src.main.python.ImageCropper import ImageCropper
from src.main.python.algorithms.noise_image import gaussian_additive_noise, rayleigh_multiplicative_noise, \
    exponential_multiplicative_noise, salt_and_pepper
from src.main.python.algorithms.operations_between_images import equalize_histogram, dynamic_range_compression, \
    gamma_power_function
from src.main.python.utils.ImageWrapper import ImageWrapper, is_raw
from src.main.python.views.OperationsBetweenImages import OperationsBetweenImages

import src.main.python.algorithms.channel_operations as op

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'ATI'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

        self.image = None

        self.imageVisualizerWindows = []
        self.views = []
        self.popUpWindow = None
        self.widgetOnSelection = None

        my_config.MainWindowSelf = self

    def initUI(self):
        with open("../css/style.css") as f:
            self.setStyleSheet(f.read())

        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        mainLayout = QVBoxLayout()

        # SELECT FILE

        imagePreviewAndDataLayout = QHBoxLayout()

        self.imageLabel = QLabel(alignment=(Qt.AlignVCenter | Qt.AlignHCenter))
        self.imageLabel.setFixedSize(600, 500)
        self.imageLabel.setStyleSheet("QLabel { border-style: solid; border-width: 2px; border-color: rgba(0, 0, 0, 0.1); }");
        imagePreviewAndDataLayout.addWidget(self.imageLabel)

        imageDataLayout = QVBoxLayout()
        imageDataLayout.setAlignment(Qt.AlignTop)

        fileLabelLayout = QHBoxLayout()
        fileLabelLayout.addWidget(QLabel("File name: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileNameLabel = QLabel("None", alignment=Qt.AlignRight)
        fileLabelLayout.addWidget(self.fileNameLabel)
        imageDataLayout.addLayout(fileLabelLayout)

        fileDirLayout = QHBoxLayout()
        fileDirLayout.addWidget(QLabel("File path: ", objectName='title', alignment=Qt.AlignLeft))
        self.filePathLabel = QLabel("None", alignment=Qt.AlignRight)
        fileDirLayout.addWidget(self.filePathLabel)
        imageDataLayout.addLayout(fileDirLayout)

        fileHeightLayout = QHBoxLayout()
        fileHeightLayout.addWidget(QLabel("Height: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileHeightLabel = QLabel("None", alignment=Qt.AlignRight)
        fileHeightLayout.addWidget(self.fileHeightLabel)
        imageDataLayout.addLayout(fileHeightLayout)

        fileWidthLayout = QHBoxLayout()
        fileWidthLayout.addWidget(QLabel("Width: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileWidthLabel = QLabel("None", alignment=Qt.AlignRight)
        fileWidthLayout.addWidget(self.fileWidthLabel)
        imageDataLayout.addLayout(fileWidthLayout)

        fileLayersLayout = QHBoxLayout()
        fileLayersLayout.addWidget(QLabel("Number of channels: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileLayersLabel = QLabel("None", alignment=Qt.AlignRight)
        fileLayersLayout.addWidget(self.fileLayersLabel)
        imageDataLayout.addLayout(fileLayersLayout)

        fileActionsLayout = QVBoxLayout()
        fileActionsLayout.setAlignment(Qt.AlignBottom)
        fileActionsLayout.addWidget(QPushButton("Change selected file", clicked=self.selectFileButton_clicked))
        fileActionsLayout.addWidget(QPushButton("Visualize and crop selected image", clicked=self.imageVisualizer_clicked))
        fileActionsLayout.addWidget(QPushButton("Open in OS image viewer", clicked=self.open_file_clicked))
        fileActionsLayout.addWidget(QPushButton("Grey level histogram", clicked=self.histogram_transformation_clicked))
        imageDataLayout.addLayout(fileActionsLayout)

        imagePreviewAndDataLayout.addLayout(imageDataLayout)


        mainLayout.addLayout(imagePreviewAndDataLayout)
        # SELECT FILE

        # ALGORITHMS
        self.tabLayout = QTabWidget()
        pointOperationTab = QWidget()
        transformationTab = QWidget()
        noiseTab = QWidget()
        filterTab = QWidget()

        algorithmsLayout = QHBoxLayout()

        algorithm2Button = QPushButton("Operations between Images")
        algorithm2Button.clicked.connect(self.operation_between_images)
        algorithmsLayout.addWidget(algorithm2Button)

        algorithm4Button = QPushButton("Equalization")
        algorithm4Button.clicked.connect(self.equalization)
        algorithmsLayout.addWidget(algorithm4Button)

        algorithm5Button = QPushButton("Dynamic range compression")
        algorithm5Button.clicked.connect(self.dynamic_range_compression_clicked)
        algorithmsLayout.addWidget(algorithm5Button)

        algorithm6Button = QPushButton("Gamma power function")
        algorithm6Button.clicked.connect(self.gamma_power_function_clicked)
        algorithmsLayout.addWidget(algorithm6Button)

        # algorithmsLayout.setEnabled(False)
        # mainLayout.addLayout(algorithmsLayout)
        pointOperationTab.setLayout(algorithmsLayout)
        self.tabLayout.addTab(pointOperationTab, "Point operations")

        # TRANSFORMATIONS
        transformationLayoutt = QHBoxLayout()

        contrastButton = QPushButton("Contrast increment")
        contrastButton.clicked.connect(self.contrast_transformation_clicked)
        transformationLayoutt.addWidget(contrastButton)

        thresholdButton = QPushButton("Threshold")
        thresholdButton.clicked.connect(self.threshold_transformation_clicked)
        transformationLayoutt.addWidget(thresholdButton)

        negativeButton = QPushButton("Negative")
        negativeButton.clicked.connect(self.negative_transformation_clicked)
        transformationLayoutt.addWidget(negativeButton)

        # mainLayout.addLayout(transformationLayoutt)
        transformationTab.setLayout(transformationLayoutt)
        self.tabLayout.addTab(transformationTab, "Transformations")

        # Noises
        noiseLayout = QHBoxLayout()

        gaussian_button = QPushButton("Gaussian additive noise")
        gaussian_button.clicked.connect(self.gaussian_noise_clicked)
        noiseLayout.addWidget(gaussian_button)

        rayleigh_button = QPushButton("Rayleigh multiplicative noise")
        rayleigh_button.clicked.connect(self.rayleigh_noise_clicked)
        noiseLayout.addWidget(rayleigh_button)

        exponential_button = QPushButton("Exponential additive noise")
        exponential_button.clicked.connect(self.exponential_noise_clicked)
        noiseLayout.addWidget(exponential_button)

        salt_and_pepper_button = QPushButton("Salt and pepper noise")
        salt_and_pepper_button.clicked.connect(self.salt_and_pepper_clicked)
        noiseLayout.addWidget(salt_and_pepper_button)

        # mainLayout.addLayout(noiseLayout)
        noiseTab.setLayout(noiseLayout)
        self.tabLayout.addTab(noiseTab, "Noises")

        # Filters
        filterLayout = QHBoxLayout()

        mean_filter_button = QPushButton("Mean filter")
        mean_filter_button.clicked.connect(self.mean_filter_clicked)
        filterLayout.addWidget(mean_filter_button)

        median_filter_button = QPushButton("Median filter")
        median_filter_button.clicked.connect(self.median_filter_clicked)
        filterLayout.addWidget(median_filter_button)

        ponderated_median_filter_button = QPushButton("Ponderated (3x3) median filter")
        ponderated_median_filter_button.clicked.connect(self.ponderated_median_filter_clicked)
        filterLayout.addWidget(ponderated_median_filter_button)

        gaussian_filter_button = QPushButton("Gaussian filter")
        gaussian_filter_button.clicked.connect(self.gaussian_filter_clicked)
        filterLayout.addWidget(gaussian_filter_button)

        highpass_filter_button = QPushButton("Highpass filter")
        highpass_filter_button.clicked.connect(self.highpass_filter_clicked)
        filterLayout.addWidget(highpass_filter_button)

        # mainLayout.addLayout(filterLayout)
        filterTab.setLayout(filterLayout)
        self.tabLayout.addTab(filterTab, "Filters")
        # ALGORITHMS

        self.tabLayout.setEnabled(False)
        mainLayout.addWidget(self.tabLayout)

        self.setLayout(mainLayout)
        self.show()

    def open_file_clicked(self):
        self.image.image_element.show()

    # La convierto a RGBA para mostrarla
    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def selectFileButton_clicked(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select image file", "",
                                                  "Images (*.jpg *.raw *.ppm *.pgm *.RAW)", options=options)
        if filePath:
            self.loadedImage_changed(ImageWrapper.from_path(filePath))
            return True

        return False

    def loadedImage_changed(self, img: ImageWrapper = None):
        if img is not None:
            self.image = img

            self.fileWidthLabel.setText(str(img.image_element.width))
            self.fileHeightLabel.setText(str(img.image_element.height))
            self.fileNameLabel.setText(img.filename)
            self.filePathLabel.setText(img.file_path)
            self.fileLayersLabel.setText(str(len(img.image_element.getbands())))

        qim = ImageQt(self.image.image_element)
        pixmap = QPixmap.fromImage(qim).scaled(self.imageLabel.width(), self.imageLabel.height(),
                                               QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.tabLayout.setEnabled(True)

    def imageVisualizer_clicked(self):
        if self.image is not None:
            new_image_window = ImageCropper(self.image)
            self.imageVisualizerWindows.append(new_image_window)
            new_image_window.show()
            # new_image_window.subscribe_selection_event(self.on_selection)
            # new_image_window.subscribe_paste_event(self.on_paste)

    def on_selection(self, widget):
        self.widgetOnSelection = widget
        for image_window in self.imageVisualizerWindows:
            if image_window != widget:
                image_window.reset_selection()

    def on_paste(self, widget):
        start, end = self.widgetOnSelection.get_selection()
        image_to_paste = self.widgetOnSelection.get_image_array()[min([start[1], end[1]]):max([start[1], end[1]]),
                         min([start[0], end[0]]):max([start[0], end[0]])]
        widget.overlap_image(image_to_paste, widget.get_last_mouse_pos())

    def operation_between_images(self):
        self.views.append(OperationsBetweenImages())

    def equalization(self):
        result = equalize_histogram(self.image)
        self.show_result(result)

    def gaussian_noise_clicked(self):
        mu, _ = QInputDialog.getDouble(self, "Select mu (expected value)", "mu", 0)
        sigma, _ = QInputDialog.getDouble(self, "Select sigma (standard deviation)", "sigma", 1)
        percentage, _ = QInputDialog.getDouble(self, "Contamination percentage", "percentage", 1)
        result = gaussian_additive_noise(self.image, mu, sigma, percentage)
        self.show_result(result)

    def rayleigh_noise_clicked(self):
        gamma, _ = QInputDialog.getDouble(self, "Select gamma (expected value)", "gamma", 0)
        percentage, _ = QInputDialog.getDouble(self, "Contamination percentage", "percentage", 1)
        result = rayleigh_multiplicative_noise(self.image, 1, percentage)
        self.show_result(result)

    def exponential_noise_clicked(self):
        _lambda, _ = QInputDialog.getDouble(self, "Select lambda", "lambda", 1)
        percentage, _ = QInputDialog.getDouble(self, "Contamination percentage", "percentage", 1)
        result = exponential_multiplicative_noise(self.image, _lambda, percentage)
        self.show_result(result)

    def salt_and_pepper_clicked(self):
        p0, _ = QInputDialog.getDouble(self, "p0", "p0", 0.1)
        p1, _ = QInputDialog.getDouble(self, "p1", "p1", 0.9)
        percentage, _ = QInputDialog.getDouble(self, "Contamination percentage", "percentage", 1)
        result = salt_and_pepper(self.image, p0, p1, percentage)
        self.show_result(result)

    def dynamic_range_compression_clicked(self):
        result = dynamic_range_compression(self.image)
        self.show_result(result)

    def gamma_power_function_clicked(self):
        gamma, _ = QInputDialog.getDouble(self, "gamma", "gamma", 0.1)
        result = gamma_power_function(self.image, gamma)
        self.show_result(result)

    def show_result(self, result: ImageWrapper):
        new_image_window = ImageCropper(result)
        self.imageVisualizerWindows.append(new_image_window)
        new_image_window.show()

    def mean_filter_clicked(self):
        window_size, _ = QInputDialog.getInt(self, "Select window size", "window size", 3)
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_mean_window(channel, window_size)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        self.show_result(img_cpy)

    def median_filter_clicked(self):
        window_size, _ = QInputDialog.getInt(self, "Select window size", "window size", 3)
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_median_window(channel, window_size)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        self.show_result(img_cpy)

    def highpass_filter_clicked(self):
        window_size, _ = QInputDialog.getInt(self, "Select window size", "window size", 3)
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_highpass_window(channel, window_size)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        self.show_result(img_cpy)

    def gaussian_filter_clicked(self):
        sigma, _ = QInputDialog.getDouble(self, "Select sigma (standard deviation)", "sigma", 1)
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_gaussian_window(channel, sigma)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        self.show_result(img_cpy)

    def ponderated_median_filter_clicked(self):
        img_cpy = self.image.copy()
        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_ponderated_median_window_3x3(channel)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))
        self.show_result(img_cpy)

    def histogram_transformation_clicked(self):
        img_cpy = self.image.copy()
        op.channel_histogram(img_cpy.image_element, True)

    def negative_transformation_clicked(self):
        img_cpy: ImageWrapper
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_negative(channel)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        self.show_result(img_cpy)

    def threshold_transformation_clicked(self):
        threshold, _ = QInputDialog.getInt(self, "Select threshold", "threshold", 125)

        img_cpy: ImageWrapper
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_threshold(channel, threshold)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        self.show_result(img_cpy)

    def contrast_transformation_clicked(self):
        K, _ = QInputDialog.getInt(self, "Select K", "K", 2)

        img_cpy: ImageWrapper
        img_cpy = self.image.copy()

        channels = img_cpy.image_element.split()
        for channel in channels:
            op.channel_contrast(channel, K)
        img_cpy.set_pillow_image(Image.merge(img_cpy.image_element.mode, channels))

        op.channel_contrast(img_cpy.image_element, K)

        self.show_result(img_cpy)

    def askForInt(self, message: str, default: int):
        intVal, _ = QInputDialog.getInt(self, message, "Enter integer value", default)
        return intVal


def main():
    my_config.initialize()
    app = QApplication(sys.argv)
    ex = MainWindow()

    #Force load file
    if not ex.selectFileButton_clicked():
        return

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
