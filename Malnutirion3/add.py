# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'add.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_AdNewAdvertizer(object):
    def setupUi(self, AdNewAdvertizer):
        AdNewAdvertizer.setObjectName(_fromUtf8("AdNewAdvertizer"))
        AdNewAdvertizer.resize(394, 383)
        self.centralwidget = QtGui.QWidget(AdNewAdvertizer)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 30, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 90, 68, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 140, 68, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 180, 68, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 220, 121, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 320, 99, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_3 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(270, 320, 99, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(180, 80, 191, 31))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.lineEdit_2 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(180, 120, 191, 31))
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.lineEdit_3 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(180, 170, 191, 31))
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.lineEdit_4 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(180, 210, 191, 31))
        self.lineEdit_4.setObjectName(_fromUtf8("lineEdit_4"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(40, 270, 111, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.lineEdit_5 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(180, 260, 191, 31))
        self.lineEdit_5.setEchoMode(QtGui.QLineEdit.Password)
        self.lineEdit_5.setObjectName(_fromUtf8("lineEdit_5"))
        AdNewAdvertizer.setCentralWidget(self.centralwidget)

        self.retranslateUi(AdNewAdvertizer)
        QtCore.QMetaObject.connectSlotsByName(AdNewAdvertizer)

    def retranslateUi(self, AdNewAdvertizer):
        AdNewAdvertizer.setWindowTitle(_translate("AdNewAdvertizer", "MainWindow", None))
        self.label.setText(_translate("AdNewAdvertizer", "Register New User", None))
        self.label_2.setText(_translate("AdNewAdvertizer", "Name", None))
        self.label_3.setText(_translate("AdNewAdvertizer", "Email", None))
        self.label_4.setText(_translate("AdNewAdvertizer", "Contact", None))
        self.label_5.setText(_translate("AdNewAdvertizer", "Username", None))
        self.pushButton.setText(_translate("AdNewAdvertizer", "Save", None))
        self.pushButton_3.setText(_translate("AdNewAdvertizer", "Cancel", None))
        self.label_6.setText(_translate("AdNewAdvertizer", "Password", None))

