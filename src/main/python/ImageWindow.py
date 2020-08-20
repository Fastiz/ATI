from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtWidgets import QWidget, QLabel

from src.main.python.utils.Image import load_image_array


def get_qImage(image_rgb):
    h, w, c = image_rgb.shape

    return QImage(image_rgb.data, h, w, 3*w, QImage.Format_RGB888)


class ImageWindow(QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.image_array = load_image_array(file_path)
        self.qImage = get_qImage(self.image_array)
        self.pressedPos = None
        self.lastMousePos = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.file_path)

        self.resize(self.qImage.width(), self.qImage.height())
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw_image(qp)
        self.draw_selection(qp)
        qp.end()

    def draw_image(self, qp):
        qp.drawImage(0, 0, self.qImage)

    def draw_selection(self, qp):
        if self.pressedPos is None or self.lastMousePos is None:
            return

        sx, sy = self.pressedPos
        ex, ey = self.lastMousePos

        qp.drawRect(min([sx, ex]), min([sy, ey]), abs(sx-ex), abs(sy-ey))

    def mousePressEvent(self, event):
        self.pressedPos = (event.x(), event.y())

    def mouseReleaseEvent(self, event):
        start = self.pressedPos
        end = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        self.lastMousePos = (event.x(), event.y())

        if self.pressedPos is not None:
            self.update()

