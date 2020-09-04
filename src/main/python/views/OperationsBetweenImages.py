from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QInputDialog

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
        self.scalar_multiplication_input = None
        self.scalar_multiplication_btn = None
        self.result_widget = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operation between two images")

        layout = QVBoxLayout()

        first_layout = QHBoxLayout()

        self.select_images_btn = QPushButton("Select images")
        self.select_images_btn.clicked.connect(self.select_images)

        first_layout.addWidget(self.select_images_btn)

        layout.addLayout(first_layout)

        second_layout = QHBoxLayout()

        self.product_btn = QPushButton("Product")
        self.product_btn.clicked.connect(self.product)

        second_layout.addWidget(self.product_btn)

        self.addition_btn = QPushButton("Addition")
        self.addition_btn.clicked.connect(self.addition)

        second_layout.addWidget(self.addition_btn)

        self.subtraction_btn = QPushButton("Subtraction")
        self.subtraction_btn.clicked.connect(self.subtraction)

        second_layout.addWidget(self.subtraction_btn)

        layout.addLayout(second_layout)

        third_layout = QHBoxLayout()

        self.scalar_multiplication_btn = QPushButton("Scalar multiplication")
        self.scalar_multiplication_btn.clicked.connect(self.scalar_multiplication)
        third_layout.addWidget(self.scalar_multiplication_btn)

        layout.addLayout(third_layout)

        self.setLayout(layout)
        self.show()

        self.set_enabled_operations(False)

    def select_images(self):
        def handler(paths, dimensions=None):
            if dimensions is None:
                dimensions = [None, None]
            self.first_image = ImageWrapper.from_path(paths[0])
            self.second_image = ImageWrapper.from_path(paths[1])
            self.set_enabled_operations(True)
            self.select_images_btn.setEnabled(False)

        MultipleImageSelector(["First image", "Second image"], "Submit",
                              "Images selection", handler)

    def set_enabled_operations(self, enabled):
        for btn in [self.product_btn, self.subtraction_btn, self.addition_btn, self.scalar_multiplication_btn]:
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

    def scalar_multiplication(self):
        scalar, _ = QInputDialog.getDouble(self, "Select scalar", "Scalar", 0)
        self.result_widget = ImageSectionSelector(operations_between_images.scalar_multiplication(self.first_image, scalar))
        self.result_widget.show()
