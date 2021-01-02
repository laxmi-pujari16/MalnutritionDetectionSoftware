# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'err_img.ui'
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

class Ui_Error(object):
    def setupUi(self, Error):
        Error.setObjectName(_fromUtf8("Error"))
        Error.resize(258, 137)
        self.label = QtGui.QLabel(Error)
        self.label.setGeometry(QtCore.QRect(40, 50, 321, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.pushButton = QtGui.QPushButton(Error)
        self.pushButton.setGeometry(QtCore.QRect(150, 90, 81, 31))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))

        self.retranslateUi(Error)
        QtCore.QMetaObject.connectSlotsByName(Error)

    def retranslateUi(self, Error):
        Error.setWindowTitle(_translate("Error", "Dialog", None))
        self.label.setText(_translate("Error", "Error Reading Image", None))
        self.pushButton.setText(_translate("Error", "OK", None))

