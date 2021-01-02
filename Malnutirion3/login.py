# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
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

class Ui_UserLogin(object):
    def setupUi(self, UserLogin):
        UserLogin.setObjectName(_fromUtf8("UserLogin"))
        UserLogin.resize(586, 305)
        self.centralwidget = QtGui.QWidget(UserLogin)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 40, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(280, 110, 101, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 170, 101, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(410, 100, 141, 27))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.lineEdit_2 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(410, 160, 141, 27))
        self.lineEdit_2.setEchoMode(QtGui.QLineEdit.Password)
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(280, 220, 99, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 220, 99, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.graphicsView = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(10, 30, 261, 261))
        self.graphicsView.setStyleSheet(_fromUtf8("background-image: url(:/newPrefix/login.png);"))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.pushButton_3 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(380, 220, 99, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        UserLogin.setCentralWidget(self.centralwidget)

        self.retranslateUi(UserLogin)
        QtCore.QMetaObject.connectSlotsByName(UserLogin)

    def retranslateUi(self, UserLogin):
        UserLogin.setWindowTitle(_translate("UserLogin", "MainWindow", None))
        self.label.setText(_translate("UserLogin", "Login", None))
        self.label_2.setText(_translate("UserLogin", "Username", None))
        self.label_3.setText(_translate("UserLogin", "Password", None))
        self.pushButton.setText(_translate("UserLogin", "Login", None))
        self.pushButton_2.setText(_translate("UserLogin", "Cancel", None))
        self.pushButton_3.setText(_translate("UserLogin", "SignUp", None))

import log_rc
