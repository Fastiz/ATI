import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QPushButton, QInputDialog

from src.main.python.ImageCropper import ImageCropper
from src.main.python.algorithms.noise_image import gaussian_additive_noise, rayleigh_multiplicative_noise, \
    exponential_multiplicative_noise
from src.main.python.algorithms.operations_between_images import equalize_histogram
from src.main.python.components.ImageSectionSelector import ImageSectionSelector
from src.main.python.components.MultipleImageSelector import MultipleImageSelector
from src.main.python.utils.ImageWrapper import ImageWrapper
from src.main.python.views.OperationsBetweenImages import OperationsBetweenImages


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

        algorithm3Button = QPushButton("Contrast (K = 2)")
        self.algorithmsLayout.addWidget(algorithm3Button)

        algorithm4Button = QPushButton("Equalization")
        algorithm4Button.clicked.connect(self.equalization)
        self.algorithmsLayout.addWidget(algorithm4Button)

        self.algorithmsLayout.setEnabled(False)
        mainLayout.addLayout(self.algorithmsLayout)

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

        mainLayout.addLayout(self.noiseLayout)

        # ALGORITHMS

        # self.openFileNameDialog()
        # self.openFileNamesDialog()
        # self.saveFileDialog()

        self.setLayout(mainLayout)
        self.show()

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

    def show_result(self, result: ImageWrapper):
        new_image_window = ImageCropper(result)
        self.imageVisualizerWindows.append(new_image_window)
        new_image_window.show()


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
