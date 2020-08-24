import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QPushButton

from src.main.python.ImageCropper import ImageCropper
from src.main.python.utils.ImageWrapper import ImageWrapper


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'ATI'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

        self.selectedFilePath = ""
        self.image = None

        self.imageVisualizerWindows = []
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

        algorithm2Button = QPushButton("Algorithm #2")
        self.algorithmsLayout.addWidget(algorithm2Button)

        algorithm3Button = QPushButton("Algorithm #3")
        self.algorithmsLayout.addWidget(algorithm3Button)

        self.algorithmsLayout.setEnabled(False)
        mainLayout.addLayout(self.algorithmsLayout)
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
        self.image = ImageWrapper(self.selectedFilePath)

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


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
