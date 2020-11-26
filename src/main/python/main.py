import math
import sys
import time
import threading

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QPushButton, QInputDialog, QTabWidget, QMessageBox

import src.main.python.algorithms.border_detection as bd
from src.main.python.components.TwoImageSelector import TwoImagesSelector

from src.main.python import my_config
from src.main.python.ImageCropper import ImageCropper, ImageSectionSelectorWindow, ImageCarrousel
from src.main.python.algorithms.bilateral_filter import bilateral_filter
from src.main.python.algorithms.border_tracking import BorderTracking
from src.main.python.algorithms.canny_border_detection import canny_border_detection
from src.main.python.algorithms.susan_border_detection import apply_susan_border_detection
from src.main.python.algorithms.diffusion import isotropic_diffusion_step, anisotropic_diffusion_step
from src.main.python.algorithms.noise_image import gaussian_additive_noise, rayleigh_multiplicative_noise, \
    exponential_multiplicative_noise, salt_and_pepper
from src.main.python.algorithms.operations_between_images import equalize_histogram, dynamic_range_compression, \
    gamma_power_function
from src.main.python.algorithms.thresholding import global_thresholding, otsu_method
from src.main.python.utils.ImageWrapper import ImageWrapper, is_raw
from src.main.python.views.OperationsBetweenImages import OperationsBetweenImages

import src.main.python.algorithms.channel_operations as op

import numpy as np


class MainWindow(QWidget):
    image: ImageWrapper

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
        self.imageLabel.setStyleSheet(
            "QLabel { border-style: solid; border-width: 2px; border-color: rgba(0, 0, 0, 0.1); }");
        imagePreviewAndDataLayout.addWidget(self.imageLabel)

        imageDataLayout = QVBoxLayout()

        imageLabelDataLayout = QVBoxLayout()
        imageLabelDataLayout.setAlignment(Qt.AlignTop)

        fileLabelLayout = QHBoxLayout()
        fileLabelLayout.addWidget(QLabel("File name: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileNameLabel = QLabel("None", alignment=Qt.AlignRight)
        fileLabelLayout.addWidget(self.fileNameLabel)
        imageLabelDataLayout.addLayout(fileLabelLayout)

        fileDirLayout = QHBoxLayout()
        fileDirLayout.addWidget(QLabel("File path: ", objectName='title', alignment=Qt.AlignLeft))
        self.filePathLabel = QLabel("None", alignment=Qt.AlignRight)
        fileDirLayout.addWidget(self.filePathLabel)
        imageLabelDataLayout.addLayout(fileDirLayout)

        fileHeightLayout = QHBoxLayout()
        fileHeightLayout.addWidget(QLabel("Height: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileHeightLabel = QLabel("None", alignment=Qt.AlignRight)
        fileHeightLayout.addWidget(self.fileHeightLabel)
        imageLabelDataLayout.addLayout(fileHeightLayout)

        fileWidthLayout = QHBoxLayout()
        fileWidthLayout.addWidget(QLabel("Width: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileWidthLabel = QLabel("None", alignment=Qt.AlignRight)
        fileWidthLayout.addWidget(self.fileWidthLabel)
        imageLabelDataLayout.addLayout(fileWidthLayout)

        fileLayersLayout = QHBoxLayout()
        fileLayersLayout.addWidget(QLabel("Number of channels: ", objectName='title', alignment=Qt.AlignLeft))
        self.fileLayersLabel = QLabel("None", alignment=Qt.AlignRight)
        fileLayersLayout.addWidget(self.fileLayersLabel)
        imageLabelDataLayout.addLayout(fileLayersLayout)

        imageDataLayout.addLayout(imageLabelDataLayout)

        fileActionsLayout = QVBoxLayout()
        fileActionsLayout.setAlignment(Qt.AlignBottom)

        fileActionsLayout.setAlignment(Qt.AlignBottom)
        fileActionsLayout.addWidget(QPushButton("Change selected file", clicked=self.selectFileButton_clicked))

        fileActionsLayout.addWidget(
            QPushButton("Visualize and crop selected image", clicked=self.imageVisualizer_clicked))
        fileActionsLayout.addWidget(QPushButton("Open in OS image viewer", clicked=self.open_file_clicked))
        fileActionsLayout.addWidget(QPushButton("Grey level histogram", clicked=self.histogram_transformation_clicked))
        fileActionsLayout.addWidget(QPushButton("Reload from disk", clicked=self.reload_from_disk_clicked))
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
        borderDetectionTab = QWidget()
        thresholdingTab = QWidget()
        objectDetectionTab = QWidget()

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

        toGrayscaleButton = QPushButton("To Grayscale")
        toGrayscaleButton.clicked.connect(self.to_grayscale)
        transformationLayoutt.addWidget(toGrayscaleButton)

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

        isotropic_filter_button = QPushButton("Isotropic diffusion")
        isotropic_filter_button.clicked.connect(self.isotropic_diffusion_method_clicked)
        filterLayout.addWidget(isotropic_filter_button)

        anisotropic_filter_button = QPushButton("Anisotropic diffusion")
        anisotropic_filter_button.clicked.connect(self.anisotropic_diffusion_method_clicked)
        filterLayout.addWidget(anisotropic_filter_button)

        bilateral_filter_button = QPushButton("Bilateral filter")
        bilateral_filter_button.clicked.connect(self.bilateral_filter_clicked)
        filterLayout.addWidget(bilateral_filter_button)

        # mainLayout.addLayout(filterLayout)
        filterTab.setLayout(filterLayout)
        self.tabLayout.addTab(filterTab, "Filters")

        # Border detection
        borderDetectionLayout = QHBoxLayout()

        borderDetectionLayout.addWidget(QPushButton("Prewitt", clicked=self.prewitt_border_detection_clicked))
        borderDetectionLayout.addWidget(QPushButton("Sobel", clicked=self.sobel_border_detection_clicked))
        borderDetectionLayout.addWidget(QPushButton("Laplacian", clicked=self.laplace_border_detection_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Laplacian (Threshold)", clicked=self.laplace_threshold_border_detection_clicked))
        borderDetectionLayout.addWidget(QPushButton("Laplacian of Gaussian", clicked=self.log_border_detection_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Generic directional operator", clicked=self.generic_derivatives_border_detection_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Canny", clicked=self.canny_border_detection_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Susan", clicked=self.susan_border_detection_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Hough transform (line detection)", clicked=self.hough_transform_line_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Hough transform (circle detection)", clicked=self.hough_transform_circunference_clicked))
        borderDetectionLayout.addWidget(
            QPushButton("Segmentation method (video)", clicked=self.selectMultipleFilesButton_clicked))

        borderDetectionTab.setLayout(borderDetectionLayout)
        self.tabLayout.addTab(borderDetectionTab, "Border detection")

        # Thresholding
        thresholdingDetectionLayout = QHBoxLayout()

        thresholdingDetectionLayout.addWidget(QPushButton("Global", clicked=self.global_thresholding))
        thresholdingDetectionLayout.addWidget(QPushButton("Otsu", clicked=self.otsu_thresholding))

        thresholdingTab.setLayout(thresholdingDetectionLayout)
        self.tabLayout.addTab(thresholdingTab, "Thresholding")




        objectDetectionLayout = QHBoxLayout()

        siftButton = QPushButton("SIFT")
        siftButton.clicked.connect(self.sift_clicked)
        objectDetectionLayout.addWidget(siftButton)

        objectDetectionTab.setLayout(objectDetectionLayout)
        self.tabLayout.addTab(objectDetectionTab, "Object detection")

        # ALGORITHMS

        mainLayout.addWidget(self.tabLayout)

        self.setLayout(mainLayout)
        self.show()

    def sift_clicked(self):
        sift_window = TwoImagesSelector(2, "SIFT")
        self.imageVisualizerWindows.append(sift_window)
        sift_window.show()

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
                                                  "Images (*.jpg *.jpeg *.raw *.ppm *.pgm *.RAW)", options=options)
        if filePath:
            self.loadedImage_changed(ImageWrapper.from_path(filePath))
            return True

        return False

    def reload_from_disk_clicked(self):
        self.loadedImage_changed(ImageWrapper.from_path(self.image.file_path))

    def loadedImage_changed(self, img: ImageWrapper = None):
        if img is not None:
            self.image = img
            self.image.draw_image()

            self.fileWidthLabel.setText(str(img.image_element.width))
            self.fileHeightLabel.setText(str(img.image_element.height))
            self.fileNameLabel.setText(img.filename)
            self.filePathLabel.setText(img.file_path)
            self.fileLayersLabel.setText(str(len(img.channels)))

        qim = ImageQt(self.image.image_element)
        pixmap = QPixmap.fromImage(qim).scaled(self.imageLabel.width(), self.imageLabel.height(),
                                               QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)

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

    def show_result(self, result: ImageWrapper, window_title: str = None):
        new_image_window = ImageCropper(result, window_title)
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

    def askForInt(self, message: str, default: int = 1, min: int = 1, max: int = 2147483647):
        intVal, _ = QInputDialog.getInt(self, "Enter integer value", message, default, min=min, max=max)
        return intVal

    def askForFloat(self, message: str, default: float = 1.0):
        floatVal, _ = QInputDialog.getDouble(self, "Enter float value", message, default)
        return floatVal

    def prewitt_border_detection_clicked(self):
        prewitt_x_mask = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])

        prewitt_y_mask = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ])

        img_cpy: ImageWrapper = self.image.copy()
        for channel in img_cpy.channels:
            bd.first_derivative_border_detection(channel, [prewitt_x_mask, prewitt_y_mask])
        self.show_result(img_cpy)

    def sobel_border_detection_clicked(self):
        sobel_x_mask = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        sobel_y_mask = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        img_cpy: ImageWrapper = self.image.copy()
        for channel in img_cpy.channels:
            bd.first_derivative_border_detection(channel, [sobel_x_mask, sobel_y_mask])
        self.show_result(img_cpy)

    def generic_derivatives_border_detection_clicked(self):
        derivative_operators_dict = dict()
        derivative_operators_dict["Prewitt"] = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])
        derivative_operators_dict["Unnamed"] = np.array([
            [1, 1, 1],
            [1, -2, 1],
            [-1, -1, -1]
        ])
        derivative_operators_dict["Kirsh"] = np.array([
            [5, 5, 5],
            [-3, 0, -3],
            [-3, -3, -3]
        ])
        derivative_operators_dict["Sobel"] = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])

        message = ""
        operator_list = []
        for operator in derivative_operators_dict.keys():
            operator_list.append(operator)
            message += str(len(operator_list)) + ": " + operator + " operator\n"

        selected_operator = self.askForInt(message, 1, 1, len(operator_list)) - 1

        selected_operator_name = operator_list[selected_operator]
        mask = derivative_operators_dict[selected_operator_name]

        for i in range(4):
            img_cpy: ImageWrapper = self.image.copy()
            for channel in img_cpy.channels:
                bd.first_derivative_border_detection(channel, [mask])
            self.show_result(img_cpy, selected_operator_name + " operator (Rotated " + str(i * 45) + "º)")
            print(mask)
            bd.rotate_matrix(mask)

    def laplace_border_detection_clicked(self):
        img_cpy: ImageWrapper = self.image.copy()

        laplace_mask = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

        for channel in img_cpy.channels:
            bd.laplace_border_detection(channel, laplace_mask)
        self.show_result(img_cpy)

    def laplace_threshold_border_detection_clicked(self):
        threshold = self.askForFloat("Threshold")

        img_cpy: ImageWrapper = self.image.copy()

        laplace_mask = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

        for channel in img_cpy.channels:
            bd.laplace_border_detection(channel, laplace_mask, threshold)
        self.show_result(img_cpy)

    def log_border_detection_clicked(self):
        sigma = self.askForInt("Sigma", 1)
        mask_size = self.askForInt("Mask size", 7)
        img_cpy: ImageWrapper = self.image.copy()

        laplace_mask = bd.generate_log_mask(mask_size, sigma)

        for channel in img_cpy.channels:
            bd.laplace_border_detection(channel, laplace_mask)
        self.show_result(img_cpy)

    def canny_border_detection_clicked(self):
        gauss_times, _ = QInputDialog.getInt(self, "Amount of gauss filters", "Amount of gauss filters", 1)

        sigmas = []
        for i in range(gauss_times):
            sigma, _ = QInputDialog.getDouble(self, "Select sigma (standard deviation)", "sigma " + str(i), 1)
            sigmas.append(sigma)

        # sigma, _ = QInputDialog.getDouble(self, "Select sigma (standard deviation)", "sigma", 1)
        t1, _ = QInputDialog.getDouble(self, "Select weak threshold", "weak", 25)
        t2, _ = QInputDialog.getDouble(self, "Select strong threshold", "strong", 50)

        image = canny_border_detection(self.image, t1, t2, sigmas)

        image.draw_image()

        self.show_result(image)

    def susan_border_detection_clicked(self):
        t, _ = QInputDialog.getInt(self, "t", "t", 27)
        s1, _ = QInputDialog.getDouble(self, "s border", "s border", 0.25, decimals=3)
        s2, _ = QInputDialog.getDouble(self, "s corner", "s corner", 0.55, decimals=3)
        self.show_result(apply_susan_border_detection(t, self.image, s1, s2))

    def isotropic_diffusion_method_clicked(self):
        number_of_steps, _ = QInputDialog.getInt(self, "Select number of steps", "Steps", 5)
        diffused_image = self.image
        for i in range(number_of_steps):
            diffused_image = isotropic_diffusion_step(diffused_image)
        self.show_result(diffused_image)

    def anisotropic_diffusion_method_clicked(self):
        number_of_steps, _ = QInputDialog.getInt(self, "Select number of steps", "Steps", 5)
        sigma, _ = QInputDialog.getInt(self, "Select Leclerc param", "Sigma", 1)
        diffused_image = self.image
        for i in range(number_of_steps):
            print(i)
            diffused_image = anisotropic_diffusion_step(diffused_image, sigma)
        self.show_result(diffused_image)

    def bilateral_filter_clicked(self):
        window_size, _ = QInputDialog.getInt(self, "Select window size", "Window size", 5)
        sigma_s, _ = QInputDialog.getDouble(self, "Select sigma s", "sigma s", 2)
        sigma_r, _ = QInputDialog.getDouble(self, "Select sigma r", "sigma r", 30)
        self.show_result(bilateral_filter(self.image, (window_size, window_size), sigma_s, sigma_r))

    def global_thresholding(self):
        start_t, _ = QInputDialog.getInt(self, "Select starting threshold", "Starting threshold", 100)
        stop_delta_t, _ = QInputDialog.getInt(self, "Select delta T", "Delta T", 2)
        image1, image2, t, it = global_thresholding(self.image, start_t, stop_delta_t)

        self.show_result(image2)

        QMessageBox.about(self, "About", 'With T=%s in %s iterations' % (t, it))

    def otsu_thresholding(self):
        image1, image2, t = otsu_method(self.image)

        self.show_result(image2)

        QMessageBox.about(self, "About", 'T=%s' % t)

    def to_grayscale(self):
        image2 = self.image.copy_mode('L')
        self.show_result(image2)

    # 40 20 epsilon3
    def hough_transform_line_clicked(self):
        img = self.image.copy_mode('L')

        roUpperBound = int(math.sqrt(2) * max(self.image.dimensions()))
        roLowerBound = -roUpperBound
        roIntervals = 30
        thetaUpperBound = 90
        thetaLowerBound = -thetaUpperBound
        thetaIntervals = 40
        winnerCount = 5
        epsilon = 5.0

        roLowerBound, _ = QInputDialog.getInt(self, "Ro lower bound", "Input value", roLowerBound)
        roUpperBound, _ = QInputDialog.getInt(self, "Ro upper bound", "Input value", roUpperBound)
        roIntervals, _ = QInputDialog.getInt(self, "Ro interval count", "Input value", roIntervals)

        thetaLowerBound, _ = QInputDialog.getInt(self, "Theta lower bound", "Input value", thetaLowerBound)
        thetaUpperBound, _ = QInputDialog.getInt(self, "Theta upper bound", "Input value", thetaUpperBound)
        thetaIntervals, _ = QInputDialog.getInt(self, "Theta interval count", "Input value", thetaIntervals)

        winnerCount, _ = QInputDialog.getInt(self, "Winner number", "Input value", winnerCount)

        epsilon, _ = QInputDialog.getDouble(self, "Epsilon", "Input value", epsilon)

        winners = bd.hough_transform_line(img.channels[0], roLowerBound, roUpperBound, roIntervals, thetaLowerBound,
                                          thetaUpperBound, thetaIntervals, winner_number=winnerCount, epsilon=epsilon)

        import matplotlib.pyplot as plt

        h, w = img.channels[0].shape
        for (theta, ro) in winners:
            theta = np.deg2rad(theta)
            xs, ys = [], []
            if math.sin(theta) != 0 and math.cos(theta) != 0:
                xs = np.linspace(0, w)
                ys = (-xs * math.cos(theta) + ro) / math.sin(theta)
            elif math.sin(theta) == 0:
                xs = np.full((2,), ro)
                ys = np.array([0, h])
            else:
                xs = np.linspace(0, w)
                ys = np.full((2,), ro)

            plt.plot(xs, ys, '-')
            plt.axis('off')
            plt.imshow(img.channels[0], cmap='Greys')
            plt.show()
            print("asd")



    def hough_transform_circunference_clicked(self):
        img = self.image.copy()
        h, w = img.channels[0].shape

        xLowerBound = 0
        xUpperBound = w
        xStep = 20

        yLowerBound = 0
        yUpperBound = h
        yStep = 20

        rLowerBound = 10
        rUpperBound = min(w, h)
        rStep = 20

        winnerCount = 5

        epsilon = 10.0

        xLowerBound, _ = QInputDialog.getInt(self, "X lower bound", "Input value", xLowerBound)
        xUpperBound, _ = QInputDialog.getInt(self, "X upper bound", "Input value", xUpperBound)
        xStep, _ = QInputDialog.getInt(self, "X interval count", "Input value", xStep)

        yLowerBound, _ = QInputDialog.getInt(self, "Y lower bound", "Input value", yLowerBound)
        yUpperBound, _ = QInputDialog.getInt(self, "Y upper bound", "Input value", yUpperBound)
        yStep, _ = QInputDialog.getInt(self, "Y interval count", "Input value", yStep)

        rLowerBound, _ = QInputDialog.getInt(self, "R lower bound", "Input value", rLowerBound)
        rUpperBound, _ = QInputDialog.getInt(self, "R upper bound", "Input value", rUpperBound)
        rStep, _ = QInputDialog.getInt(self, "R interval count", "Input value", rStep)

        winnerCount, _ = QInputDialog.getInt(self, "Winner number", "Input value", winnerCount)

        epsilon, _ = QInputDialog.getDouble(self, "Epsilon", "Input value", epsilon)

        img = self.image.copy()

        # 30 30 30 11 / 8 (solo)
        winners = bd.hough_transform_circunference(img.channels[0], xLowerBound, xUpperBound, xStep, yLowerBound,
                                                   yUpperBound, yStep, rLowerBound, rUpperBound, rStep, epsilon=epsilon,
                                                   winner_number=winnerCount)

        import matplotlib.pyplot as plt

        ax = plt.gca()
        for (x, y, r) in winners:
            circle1 = plt.Circle((x, y), r, color='r', fill=False)
            ax.add_artist(circle1)

        plt.axis('off')
        plt.imshow(img.channels[0], cmap='Greys')
        plt.show()

    def SIFT_clicked(self):
        """img1 = cv2.imread('eiffel_2.jpeg')
        img2 = cv2.imread('eiffel_1.jpg')

        sift = cv2.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create()"""

    def selectMultipleFilesButton_clicked(self):
        options = QFileDialog.Options(QFileDialog.ExistingFiles)
        filePaths, _ = QFileDialog.getOpenFileNames(self, "Select image files", "",
                                                    "Images (*.jpg *.jpeg *.raw *.ppm *.pgm *.RAW)", options=options)

        filePaths.sort()

        self.images_paths = filePaths

        new_image_window = ImageSectionSelectorWindow(ImageWrapper.from_path(filePaths[0]), self.prueba_callback, "Select section")
        self.imageVisualizerWindows.append(new_image_window)
        new_image_window.show()

    add_image_to_carrousel_signal = QtCore.pyqtSignal(ImageWrapper, str)
    update_progressbar_signal = QtCore.pyqtSignal(float)
    def prueba_callback(self, points, window):
        window.close()
        new_image_window = ImageCarrousel(self)
        self.imageVisualizerWindows.append(new_image_window)

        new_image_window.show()

        self.image_carroulsel_window = new_image_window

        self.add_image_to_carrousel_signal.connect(new_image_window.addImage)
        self.update_progressbar_signal.connect(new_image_window.update_progress)

        self.points = points

        epsilon, _ = QInputDialog.getDouble(self, "Epsilon", "Epsilon", 150)

        self.epsilon = epsilon

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    def run(self):
        point_a, point_b = self.points

        dim = (point_b[0] - point_a[0], point_b[1] - point_a[1])

        border_tracking = BorderTracking(point_a, dim, self.epsilon)

        for i, img_path in enumerate(self.images_paths):
            startTime = time.time()
            result = border_tracking.next_image(ImageWrapper.from_path(img_path))
            result.draw_image()
            self.add_image_to_carrousel_signal.emit(result,
                                                    f"<b>Frame {i + 1}/{len(self.images_paths)}</b> (Time spent: {round(time.time() - startTime, 2)}s)")
            self.update_progressbar_signal.emit((i+1)/len(self.images_paths) * 100)


def main():
    my_config.initialize()
    app = QApplication(sys.argv)
    ex = MainWindow()

    # Force load file
    if not ex.selectFileButton_clicked():
        return

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
