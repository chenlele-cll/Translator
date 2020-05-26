# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Translator.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from langid import classify


class Ui_MainWindow(object):
    def __init__(self,MainWindow):
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)

        self.plainTextEdit.textChanged.connect(self.textChanging_input_text)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(20, 139, 361, 391))
        self.plainTextEdit.setObjectName("input_text")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(419, 139, 361, 391))
        self.textBrowser.setObjectName("textBrowser")
        self.src = QtWidgets.QLabel(self.centralwidget)
        self.src.setGeometry(QtCore.QRect(260, 30, 72, 15))
        self.src.setObjectName("src")
        self.tgt = QtWidgets.QLabel(self.centralwidget)
        self.tgt.setGeometry(QtCore.QRect(450, 30, 72, 15))
        self.tgt.setObjectName("tgt")
        self.detect_lang = QtWidgets.QLabel(self.centralwidget)
        self.detect_lang.setGeometry(QtCore.QRect(30, 110, 72, 15))
        self.detect_lang.setObjectName("detect_lang")
        self.tran_res = QtWidgets.QLabel(self.centralwidget)
        self.tran_res.setGeometry(QtCore.QRect(700, 110, 72, 15))
        self.tran_res.setObjectName("tran_res")
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(370, 20, 47, 21))
        self.toolButton.setObjectName("toolButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.src.setText(_translate("MainWindow", "English"))
        self.tgt.setText(_translate("MainWindow", "中文"))
        self.detect_lang.setText(_translate("MainWindow", "检测语言"))
        self.tran_res.setText(_translate("MainWindow", "翻译结果"))
        self.toolButton.setText(_translate("MainWindow", "..."))

    def textChanging_input_text(self):
        input_txt = self.plainTextEdit.toPlainText()
        print(input_txt)