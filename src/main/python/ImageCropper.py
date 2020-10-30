from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

from src.main.python.components.ImageSectionSelector import ImageSectionSelector
from src.main.python.utils.ImageWrapper import ImageWrapper


class ImageCropper(QWidget):
    def __init__(self, image: ImageWrapper, window_title: str = None):
        super().__init__()
        self.image = image

        self.init_ui(window_title)

    def init_ui(self, window_title: str = None):
        width, height = self.image.dimensions()
        self.setGeometry(30, 30, width, height)
        self.setWindowTitle(self.image.file_name() if window_title is None else window_title)
        self.center()

        layout = QVBoxLayout()

        self.imageSectionSelector = ImageSectionSelector(self.image)
        self.imageSectionSelector.subscribe_selection_finished(self.on_selection_finished)
        layout.addWidget(self.imageSectionSelector)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_selection_finished(self):
        self.btn_save.setEnabled(True)

    def on_save_clicked(self):
        img = self.image.pillow_image()
        selection_start, selection_end = self.imageSectionSelector.get_selection()

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

        img_left.show()
        """cropped_img = img.crop(selection_start.x(), selection_start.y(),
                               selection_end.x(), selection_end.y()).show()"""


class ImageSectionSelectorWindow(QWidget):
    def __init__(self, image: ImageWrapper, callback_function, window_title: str = None):
        super().__init__()
        self.image = image
        self.callback_function = callback_function

        self.init_ui(window_title)

    def init_ui(self, window_title: str = None):
        width, height = self.image.dimensions()
        self.setGeometry(30, 30, width, height)
        self.setWindowTitle(self.image.file_name() if window_title is None else window_title)
        self.center()

        layout = QVBoxLayout()

        self.imageSectionSelector = ImageSectionSelector(self.image)
        self.imageSectionSelector.subscribe_selection_finished(self.on_selection_finished)
        layout.addWidget(self.imageSectionSelector)

        self.btn_save = QPushButton("Select section")
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_selection_finished(self):
        self.btn_save.setEnabled(True)

    def on_save_clicked(self):
        img = self.image.pillow_image()
        selection_start, selection_end = self.imageSectionSelector.get_selection()

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

        img_left_area = ((smaller_x, smaller_y), (bigger_x, bigger_y))

        self.callback_function(img_left_area, self)


class ImageCarrousel(QWidget):
    def __init__(self, origin):
        super().__init__()

        self.origin = origin

        origin.add_image_to_carrousel_signal.connect(self.addImage)

        self.initUI()

    def initUI(self):
        self.scroll = QScrollArea()  # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()  # Widget that contains the collection of Vertical Box
        self.hbox = QHBoxLayout()  # The Vertical Box that contains the Horizontal Boxes of  labels and buttons

        self.widget.setLayout(self.hbox)

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')

        return

    @QtCore.pyqtSlot(ImageWrapper, str)
    def addImage(self, image: ImageWrapper, caption: str = None):
        if len(self.hbox.children()) > 0:
            line = QFrame()
            line.setFrameShape(QFrame.VLine);
            line.setFrameShadow(QFrame.Sunken);
            self.hbox.addWidget(line)

        container = QVBoxLayout()
        container.setAlignment(Qt.AlignVCenter)
        imageSectionSelector = ImageSectionSelector(image)
        container.addWidget(imageSectionSelector)
        if caption is not None:
            container.addWidget(QLabel(caption))
        self.hbox.addLayout(container)
