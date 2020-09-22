import sys

from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from PIL.ImageQt import ImageQt

from src.main.python import my_config

from src.main.python.algorithms.channel_operations import channel_threshold, channel_contrast, channel_negative, \
    channel_histogram
from src.main.python.utils.ImageWrapper import ImageWrapper

import src.main.python.algorithms.channel_operations as op

class ImageSectionSelector(QWidget):
    def __init__(self, image: ImageWrapper):
        super().__init__()
        self.image = image
        image.draw_image()

        qim = ImageQt(image.image_element)
        # self.drawn_image = QPixmap(self.image.file_path)
        # self.drawn_image = QPixmap.fromImage(qim)

        """channels = image.image_element.split()
        for channel in channels:
            op.channel_penderated_median_window_3x3(channel)
            # x = channel_histogram(channel, True)"""

        #pil_img = Image.merge(image.image_element.mode, channels)

        #self.drawn_image = self.pil2pixmap(pil_img)
        self.drawn_image = self.pil2pixmap(image.image_element)

        self.selection_start = None
        self.selection_end = None

        self.drawing = False
        self.lastPoint = QPoint()

        self.curr_pointer = None

        self.handler_selection_finished = None

        self.init_ui()

    #La convierto a RGBA para mostrarla
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

    def init_ui(self):
        width, height = self.image.dimensions()
        # self.setGeometry(0, 0, width, height)
        self.setFixedSize(width, height)
        self.setWindowTitle(self.image.file_name())
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def selection_finished(self):
        # QMessageBox.information(self, 'Image section selected', "Selection was successful")
        if self.handler_selection_finished is not None:
            self.handler_selection_finished()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.selection_start = event.pos()
        elif event.button() == Qt.RightButton:
            self.drawing = False
            self.selection_start = None
            self.selection_end = None
            self.update()
        elif event.button() == Qt.MiddleButton:

            selection_start, selection_end = self.get_selection()

            if selection_start is not None:
                newImg = self.image.copy()
                img = newImg.pillow_image()

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

                newImg.set_pillow_image(img_left)

                my_config.MainWindowSelf.loadedImage_changed(newImg)

            else:
                my_config.MainWindowSelf.loadedImage_changed(self.image)

    def mouseMoveEvent(self, event):
        self.curr_pointer = event.pos()

        width, height = self.image.dimensions()

        if event.buttons() and Qt.LeftButton and self.drawing and 0 < event.pos().x() <= width and 0 < event.pos().y() <= height:
            self.selection_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.selection_finished()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.drawn_image)

        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
        painter.setFont(QFont('Decorative', 20))

        # if self.curr_pointer is not None:
        #     painter.drawText(0, 40, "(" + str(self.curr_pointer.x()) + ", " + str(self.curr_pointer.y()) + ")")

        if self.selection_start is not None and self.selection_end is not None:
            painter.drawRect(self.selection_start.x(), self.selection_start.y(),
                             self.selection_end.x() - self.selection_start.x(),
                             self.selection_end.y() - self.selection_start.y())

    # Public methods START
    def get_selection(self):
        return self.selection_start, self.selection_end

    def subscribe_selection_finished(self, handler):
        self.handler_selection_finished = handler
    # Public methods END

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F1:
            my_config.MainWindowSelf.loadedImage_changed(self.image)

