from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

from src.main.python.algorithms import operations_between_images
from src.main.python.components.ImageSectionSelector import ImageSectionSelector
from src.main.python.components.MultipleImageSelector import MultipleImageSelector
from src.main.python.utils.ImageWrapper import ImageWrapper


class OperationsBetweenImages(QWidget):
    def __init__(self):
        super().__init__()

        self.select_images_btn = None
        self.product_btn = None
        self.addition_btn = None
        self.subtraction_btn = None
        self.first_image = None
        self.second_image = None

        self.result_widget = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operation between two images")

        layout = QVBoxLayout()

        first_layout = QHBoxLayout()

        select_images_btn = QPushButton("Select images")
        select_images_btn.clicked.connect(self.select_images)

        self.select_images_btn = select_images_btn

        first_layout.addWidget(select_images_btn)

        layout.addLayout(first_layout)

        second_layout = QHBoxLayout()

        product_btn = QPushButton("Product")
        product_btn.clicked.connect(self.product)

        self.product_btn = product_btn

        second_layout.addWidget(product_btn)

        addition_btn = QPushButton("Addition")
        addition_btn.clicked.connect(self.addition)

        self.addition_btn = addition_btn

        second_layout.addWidget(addition_btn)

        subtraction_btn = QPushButton("Subtraction")
        subtraction_btn.clicked.connect(self.subtraction)

        self.subtraction_btn = subtraction_btn

        second_layout.addWidget(subtraction_btn)

        layout.addLayout(second_layout)

        self.setLayout(layout)
        self.show()

        self.set_enabled_operations(False)

    def select_images(self):
        def handler(paths):
            self.first_image = ImageWrapper.from_path(paths[0])
            self.second_image = ImageWrapper.from_path(paths[1])
            self.set_enabled_operations(True)
            self.select_images_btn.setEnabled(False)

        MultipleImageSelector(["First image", "Second image"], "Submit",
                              "Images selection", handler)

    def set_enabled_operations(self, enabled):
        for btn in [self.product_btn, self.subtraction_btn, self.addition_btn]:
            btn.setEnabled(enabled)

    def product(self):
        self.result_widget = ImageSectionSelector(operations_between_images.product(self.first_image, self.second_image))
        self.result_widget.show()

    def addition(self):
        self.result_widget = ImageSectionSelector(operations_between_images.addition(self.first_image, self.second_image))
        self.result_widget.show()

    def subtraction(self):
        self.result_widget = ImageSectionSelector(operations_between_images.subtraction(self.first_image, self.second_image))
        self.result_widget.show()
