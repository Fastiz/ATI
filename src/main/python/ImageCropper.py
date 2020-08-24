from PyQt5.QtWidgets import *

from src.main.python.ImageSectionSelector import ImageSectionSelector
from src.main.python.utils.ImageWrapper import ImageWrapper


class ImageCropper(QWidget):
    def __init__(self, image: ImageWrapper):
        super().__init__()
        self.image = image

        self.init_ui()

    def init_ui(self):
        width, height = self.image.dimensions()
        self.setGeometry(30, 30, width, height)
        self.setWindowTitle(self.image.file_name())
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
