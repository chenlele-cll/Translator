
from Translator import Ui_MainWindow
from PyQt5.QtWidgets import QApplication,QMainWindow
from sys import argv,exit
import numpy as np


if __name__ == "__main__":

    app = QApplication(argv)
    # a = np.random.permutation(16)
    # print(a)
    # print(np.__version__)
    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())