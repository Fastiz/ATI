import os
import sys

from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QLabel, QPushButton, QGroupBox
from PyQt5.QtGui import QIcon, QPixmap


class ImageWindow(QWidget):
    def __init__(self, selectedFilePath):
        super().__init__()
        self.selectedFilePath = selectedFilePath
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Selected image visualizer")

        label = QLabel(self)
        pixmap = QPixmap(self.selectedFilePath)
        label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())


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

    def initUI(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        mainLayout = QVBoxLayout()

        # SELECT FILE
        selectedFileLayout = QVBoxLayout();

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
                                                  "Images (*.jpg *.raw *.ppm *.RAW)", options=options)
        if filePath:
            self.selectedFile_changed(filePath)

    def selectedFile_changed(self, filePath):
        self.selectedFilePath = filePath
        self.fileNameLabel.setText(os.path.basename(filePath))
        self.algorithmsLayout.setEnabled(True)

    def imageVisualizer_clicked(self):
        self.imageVisualizerWindow = ImageWindow(self.selectedFilePath)
        self.imageVisualizerWindow.show()


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
