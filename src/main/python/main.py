import os
import sys

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QPushButton, QInputDialog

from src.main.python import my_config
from src.main.python.ImageCropper import ImageCropper
from src.main.python.algorithms.noise_image import gaussian_additive_noise, rayleigh_multiplicative_noise, \
    exponential_multiplicative_noise, salt_and_pepper
from src.main.python.algorithms.operations_between_images import equalize_histogram, dynamic_range_compression, \
    gamma_power_function
from src.main.python.components.ImageSectionSelector import ImageSectionSelector
from src.main.python.components.MultipleImageSelector import MultipleImageSelector
from src.main.python.utils.ImageWrapper import ImageWrapper
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
        self.noiseLayout = None

        self.selectedFilePath = ""
        self.image = None

        self.imageVisualizerWindows = []
        self.views = []
        self.popUpWindow = None
        self.widgetOnSelection = None

        my_config.MainWindowSelf = self

    def initUI(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        mainLayout = QVBoxLayout()

        # SELECT FILE
        selectedFileLayout = QVBoxLayout()

        fileLabelLayout = QHBoxLayout()
        fileLabelLayout.addWidget(QLabel("Selected file: "))
        self.fileNameLabel = QLabel("None")
        self.fileNameLabel.setAlignment(Qt.AlignRight)
        fileLabelLayout.addWidget(self.fileNameLabel)

        selectedFileLayout.addLayout(fileLabelLayout)

        selectFileButton = QPushButton("Select from filesystem")
        selectFileButton.clicked.connect(self.selectFileButton_clicked)

        selectedFileLayout.addWidget(selectFileButton)

        self.imageLabel = QLabel()
        """pixmap = QPixmap.fromImage(self.image.image_element)
        pixmap = pixmap.scaledToWidth(20)
        self.imageLabel.setPixmap(pixmap)"""
        selectedFileLayout.addWidget(self.imageLabel)

        mainLayout.addLayout(selectedFileLayout)
        # SELECT FILE

        # ALGORITHMS
        self.algorithmsLayout = QHBoxLayout()

        algorithm1Button = QPushButton("Visualize selected image")
        algorithm1Button.clicked.connect(self.imageVisualizer_clicked)
        self.algorithmsLayout.addWidget(algorithm1Button)

        algorithm2Button = QPushButton("Operations between Images")
        algorithm2Button.clicked.connect(self.operation_between_images)
        self.algorithmsLayout.addWidget(algorithm2Button)

        algorithm4Button = QPushButton("Equalization")
        algorithm4Button.clicked.connect(self.equalization)
        self.algorithmsLayout.addWidget(algorithm4Button)

        algorithm5Button = QPushButton("Dynamic range compression")
        algorithm5Button.clicked.connect(self.dynamic_range_compression_clicked)
        self.algorithmsLayout.addWidget(algorithm5Button)

        algorithm6Button = QPushButton("Gamma power function")
        algorithm6Button.clicked.connect(self.gamma_power_function_clicked)
        self.algorithmsLayout.addWidget(algorithm6Button)

        self.algorithmsLayout.setEnabled(False)
        mainLayout.addLayout(self.algorithmsLayout)




        self.transformationLayout = QHBoxLayout()

        contrastButton = QPushButton("Contrast increment")
        contrastButton.clicked.connect(self.contrast_transformation_clicked)
        self.transformationLayout.addWidget(contrastButton)

        thresholdButton = QPushButton("Threshold")
        thresholdButton.clicked.connect(self.threshold_transformation_clicked)
        self.transformationLayout.addWidget(thresholdButton)

        negativeButton = QPushButton("Negative")
        negativeButton.clicked.connect(self.negative_transformation_clicked)
        self.transformationLayout.addWidget(negativeButton)

        histogramButton = QPushButton("Grey level histogram")
        histogramButton.clicked.connect(self.histogram_transformation_clicked)
        self.transformationLayout.addWidget(histogramButton)

        mainLayout.addLayout(self.transformationLayout)


        self.noiseLayout = QHBoxLayout()

        gaussian_button = QPushButton("Gaussian additive noise")
        gaussian_button.clicked.connect(self.gaussian_noise_clicked)
        self.noiseLayout.addWidget(gaussian_button)

        rayleigh_button = QPushButton("Rayleigh multiplicative noise")
        rayleigh_button.clicked.connect(self.rayleigh_noise_clicked)
        self.noiseLayout.addWidget(rayleigh_button)

        exponential_button = QPushButton("Exponential additive noise")
        exponential_button.clicked.connect(self.exponential_noise_clicked)
        self.noiseLayout.addWidget(exponential_button)

        salt_and_pepper_button = QPushButton("Salt and pepper noise")
        salt_and_pepper_button.clicked.connect(self.salt_and_pepper_clicked)
        self.noiseLayout.addWidget(salt_and_pepper_button)

        mainLayout.addLayout(self.noiseLayout)

        self.filterLayout = QHBoxLayout()

        mean_filter_button = QPushButton("Mean filter")
        mean_filter_button.clicked.connect(self.mean_filter_clicked)
        self.filterLayout.addWidget(mean_filter_button)

        median_filter_button = QPushButton("Median filter")
        median_filter_button.clicked.connect(self.median_filter_clicked)
        self.filterLayout.addWidget(median_filter_button)

        ponderated_median_filter_button = QPushButton("Ponderated (3x3) median filter")
        ponderated_median_filter_button.clicked.connect(self.ponderated_median_filter_clicked)
        self.filterLayout.addWidget(ponderated_median_filter_button)

        gaussian_filter_button = QPushButton("Gaussian filter")
        gaussian_filter_button.clicked.connect(self.gaussian_filter_clicked)
        self.filterLayout.addWidget(gaussian_filter_button)

        highpass_filter_button = QPushButton("Highpass filter")
        highpass_filter_button.clicked.connect(self.highpass_filter_clicked)
        self.filterLayout.addWidget(highpass_filter_button)

        mainLayout.addLayout(self.filterLayout)

        # ALGORITHMS

        # self.openFileNameDialog()
        # self.openFileNamesDialog()
        # self.saveFileDialog()

        self.setLayout(mainLayout)
        self.show()

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
        # options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Select image file", "",
                                                  "Images (*.jpg *.raw *.ppm *.pgm *.RAW)", options=options)
        if filePath:
            self.selectedFile_changed(filePath)

    def selectedFile_changed(self, filePath):
        self.selectedFilePath = filePath
        self.fileNameLabel.setText(os.path.basename(filePath))
        self.algorithmsLayout.setEnabled(True)
        self.image = ImageWrapper.from_path(self.selectedFilePath)
        self.loadedImage_changed()

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
        result = gaussian_additive_noise(self.image, mu, sigma)
        self.show_result(result)

    def rayleigh_noise_clicked(self):
        gamma, _ = QInputDialog.getDouble(self, "Select gamma (expected value)", "gamma", 0)
        result = rayleigh_multiplicative_noise(self.image, 1)
        self.show_result(result)

    def exponential_noise_clicked(self):
        _lambda, _ = QInputDialog.getDouble(self, "Select lambda", "lambda", 1)
        result = exponential_multiplicative_noise(self.image, _lambda)
        self.show_result(result)

    def salt_and_pepper_clicked(self):
        p0, _ = QInputDialog.getDouble(self, "p0", "p0", 0.1)
        p1, _ = QInputDialog.getDouble(self, "p1", "p1", 0.9)
        result = salt_and_pepper(self.image, p0, p1)
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
        op.channel_mean_window(img_cpy.image_element, window_size)
        self.show_result(img_cpy)

    def median_filter_clicked(self):
        window_size, _ = QInputDialog.getInt(self, "Select window size", "window size", 3)
        img_cpy = self.image.copy()
        op.channel_mean_window(img_cpy.image_element, window_size)
        self.show_result(img_cpy)

    def highpass_filter_clicked(self):
        window_size, _ = QInputDialog.getInt(self, "Select window size", "window size", 3)
        img_cpy = self.image.copy()
        op.channel_highpass_window(img_cpy.image_element, window_size)
        self.show_result(img_cpy)

    def gaussian_filter_clicked(self):
        sigma, _ = QInputDialog.getDouble(self, "Select sigma (standard deviation)", "sigma", 1)
        img_cpy = self.image.copy()
        op.channel_gaussian_window(img_cpy.image_element, sigma)
        self.show_result(img_cpy)

    def ponderated_median_filter_clicked(self):
        img_cpy = self.image.copy()
        op.channel_ponderated_median_window_3x3(img_cpy.image_element)
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

        op.channel_threshold(img_cpy.image_element, threshold)

        self.show_result(img_cpy)

    def contrast_transformation_clicked(self):
        K, _ = QInputDialog.getInt(self, "Select K", "K", 2)

        img_cpy: ImageWrapper
        img_cpy = self.image.copy()

        op.channel_contrast(img_cpy.image_element, K)

        self.show_result(img_cpy)

    def loadedImage_changed(self, img: ImageWrapper = None):
        if img is not None:
            self.image = img
        qim = ImageQt(self.image.image_element)
        pixmap = QPixmap.fromImage(qim)
        pixmap = pixmap.scaledToWidth(self.width)
        self.imageLabel.setPixmap(pixmap)

    def askForInt(self, message: str, default: int):
        intVal, _ = QInputDialog.getInt(self, message, "Enter integer value", default)
        return intVal

def main():
    my_config.initialize()
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
