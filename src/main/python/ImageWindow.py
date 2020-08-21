import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog

from src.main.python.utils.Image import load_image_array, save_image_array


def get_qImage(image_rgb):
    h, w, c = image_rgb.shape

    return QImage(image_rgb.data, h, w, 3*w, QImage.Format_RGB888)


class ImageWindow(QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.image_array = np.copy(load_image_array(file_path))
        self.qImage = get_qImage(self.image_array)
        self.pressedPos = None
        self.lastMousePos = None
        self.selection = None

        self.on_selection_events = set()
        self.on_paste_events = set()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.file_path)

        self.resize(self.qImage.width(), self.qImage.height() + 50)

        layout = QVBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_image)
        layout.addWidget(save_btn, alignment=Qt.AlignBottom)

        self.setLayout(layout)

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
        if event.button() == Qt.LeftButton:
            self.pressedPos = (event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.lastMousePos = (event.x(), event.y())
            self.run_paste_event()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            start = self.pressedPos
            end = (event.x(), event.y())
            self.pressedPos = None
            self.selection = (start, end)
            self.run_selection_events()

    def mouseMoveEvent(self, event):
        self.lastMousePos = (event.x(), event.y())

        if self.pressedPos is not None:
            self.update()

    def get_selection(self):
        return self.selection

    def get_image_array(self):
        return self.image_array

    def reset_selection(self):
        self.pressedPos = None
        self.update()

    def run_selection_events(self):
        for f in self.on_selection_events:
            f(self)

    def subscribe_selection_event(self, func_event):
        self.on_selection_events.add(func_event)

    def unsubscribe_selection_event(self, func_event):
        self.on_selection_events.remove(func_event)

    def subscribe_paste_event(self, func_event):
        self.on_paste_events.add(func_event)

    def unsubscribe_paste_event(self, func_event):
        self.on_paste_events.remove(func_event)

    def run_paste_event(self):
        for f in self.on_paste_events:
            f(self)

    def overlap_image(self, image, pos):
        h = image.shape[0]
        w = image.shape[1]
        x, y = pos

        length_x = min([w, self.image_array.shape[1]-x])
        length_y = min([h, self.image_array.shape[0]-y])

        self.image_array[y:y+length_y, x:x+length_x] = image[0:length_y, 0:length_x]
        self.update()

    def get_last_mouse_pos(self):
        return self.lastMousePos

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "All Files (*);;Text "
                                                                                              "Files (*.txt)")
        if file_path:
            save_image_array(file_path, self.image_array)
