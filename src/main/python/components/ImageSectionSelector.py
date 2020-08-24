import sys

from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from PIL.ImageQt import ImageQt

from src.main.python.utils.ImageWrapper import ImageWrapper


class ImageSectionSelector(QWidget):
    def __init__(self, image: ImageWrapper):
        super().__init__()
        self.image = image

        qim = ImageQt(image.image_element)
        # self.drawn_image = QPixmap(self.image.file_path)
        # self.drawn_image = QPixmap.fromImage(qim)
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
