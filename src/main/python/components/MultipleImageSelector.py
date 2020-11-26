import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QInputDialog

from src.main.python.utils.ImageWrapper import is_raw


class MultipleImageSelector(QWidget):
    def __init__(self, options_txt, submit_txt, window_title, handler, requirements=None):
        super().__init__()

        self.requirements = [True for i in range(len(options_txt))] if requirements is None else requirements

        self.handler = handler
        self.paths = [None for i in range(len(options_txt))]
        self.paths_labels = []

        self.submit_btn = None

        self.init_ui(options_txt, submit_txt, window_title)

    def init_ui(self, options_txt, submit_txt, window_title):
        self.setWindowTitle(window_title)

        main_layout = QVBoxLayout()

        for i in range(len(options_txt)):
            file_label_layout = QHBoxLayout()

            file_label_layout.addWidget(QLabel("Selected file: "))

            file_name_label = QLabel("None")
            file_name_label.setAlignment(Qt.AlignRight)
            file_label_layout.addWidget(file_name_label)
            self.paths_labels.append(file_name_label)

            select_file_button = QPushButton("Select from filesystem")
            select_file_button.clicked.connect(self.select_file_button_clicked(i))

            main_layout.addLayout(file_label_layout)
            main_layout.addWidget(select_file_button)

        submit_btn = QPushButton(submit_txt)
        submit_btn.clicked.connect(self.on_submit)
        main_layout.addWidget(submit_btn)

        self.submit_btn = submit_btn

        self.setLayout(main_layout)
        self.show()

    def check_if_submit_should_be_enabled(self):
        for i in range(len(self.requirements)):
            requirement = self.requirements[i]
            if requirement:
                if self.paths[i] is None:
                    self.submit_btn.setEnabled(False)
                    return
        self.submit_btn.setEnabled(True)

    def select_file_button_clicked(self, path_index):
        def add_path():
            options = QFileDialog.Options()
            # options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getOpenFileName(self, "Select image file", "",
                                                       "Images (*.jpg *.raw *.ppm *.pgm *.RAW *.png)", options=options)
            if file_path:

                self.paths[path_index] = file_path
                self.paths_labels[path_index].setText(file_path)

                self.check_if_submit_should_be_enabled()

        return add_path

    def on_submit(self):
        self.handler(self.paths)
        self.close()
