# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UVVIS_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
  
    
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1368, 724)
        MainWindow.setMinimumSize(QtCore.QSize(800, 550))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background:rgb(91,90,90);")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_top = QtWidgets.QFrame(self.centralwidget)
        self.frame_top.setMaximumSize(QtCore.QSize(16777215, 55))
        self.frame_top.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_top)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_top_east = QtWidgets.QFrame(self.frame_top)
        self.frame_top_east.setMaximumSize(QtCore.QSize(16777215, 55))
        self.frame_top_east.setStyleSheet("background:rgb(51,51,51);")
        self.frame_top_east.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top_east.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_top_east.setObjectName("frame_top_east")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_top_east)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_appname = QtWidgets.QFrame(self.frame_top_east)
        self.frame_appname.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_appname.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_appname.setObjectName("frame_appname")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.frame_appname)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setSpacing(7)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.lab_appname = QtWidgets.QLabel(self.frame_appname)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Light")
        font.setPointSize(24)
        self.lab_appname.setFont(font)
        self.lab_appname.setStyleSheet("color:rgb(255,255,255);")
        self.lab_appname.setObjectName("lab_appname")
        self.horizontalLayout_10.addWidget(self.lab_appname)
        self.horizontalLayout_4.addWidget(self.frame_appname)
        self.frame_user = QtWidgets.QFrame(self.frame_top_east)
        self.frame_user.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_user.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_user.setObjectName("frame_user")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame_user)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.lab_user = QtWidgets.QLabel(self.frame_user)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Light")
        font.setPointSize(24)
        self.lab_user.setFont(font)
        self.lab_user.setStyleSheet("color:rgb(255,255,255);")
        self.lab_user.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lab_user.setObjectName("lab_user")
        self.horizontalLayout_9.addWidget(self.lab_user)
        self.horizontalLayout_4.addWidget(self.frame_user)
        self.frame_person = QtWidgets.QFrame(self.frame_top_east)
        self.frame_person.setMinimumSize(QtCore.QSize(55, 55))
        self.frame_person.setMaximumSize(QtCore.QSize(55, 55))
        self.frame_person.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_person.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_person.setObjectName("frame_person")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_person)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.lab_person = QtWidgets.QLabel(self.frame_person)
        self.lab_person.setMaximumSize(QtCore.QSize(55, 55))
        self.lab_person.setText("")
        self.lab_person.setPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/peple.png"))
        self.lab_person.setScaledContents(False)
        self.lab_person.setAlignment(QtCore.Qt.AlignCenter)
        self.lab_person.setObjectName("lab_person")
        self.horizontalLayout_8.addWidget(self.lab_person)
        self.horizontalLayout_4.addWidget(self.frame_person)
        self.horizontalLayout.addWidget(self.frame_top_east)
        self.verticalLayout.addWidget(self.frame_top)
        self.frame_bottom = QtWidgets.QFrame(self.centralwidget)
        self.frame_bottom.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom.setObjectName("frame_bottom")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_bottom)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_bottom_west = QtWidgets.QFrame(self.frame_bottom)
        self.frame_bottom_west.setMinimumSize(QtCore.QSize(80, 0))
        self.frame_bottom_west.setMaximumSize(QtCore.QSize(80, 16777215))
        self.frame_bottom_west.setStyleSheet("background:rgb(51,51,51);")
        self.frame_bottom_west.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom_west.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom_west.setObjectName("frame_bottom_west")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_bottom_west)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_home = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_home.setMinimumSize(QtCore.QSize(80, 55))
        self.frame_home.setMaximumSize(QtCore.QSize(160, 55))
        self.frame_home.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_home.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_home.setObjectName("frame_home")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.frame_home)
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.bn_home = QtWidgets.QPushButton(self.frame_home)
        self.bn_home.setMinimumSize(QtCore.QSize(80, 55))
        self.bn_home.setMaximumSize(QtCore.QSize(160, 55))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.bn_home.setFont(font)
        self.bn_home.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border: none;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.bn_home.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/homeAsset 46.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bn_home.setIcon(icon)
        self.bn_home.setIconSize(QtCore.QSize(22, 22))
        self.bn_home.setFlat(True)
        self.bn_home.setObjectName("bn_home")
        self.horizontalLayout_15.addWidget(self.bn_home)
        self.verticalLayout_3.addWidget(self.frame_home)
        self.frame_bug = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_bug.setMinimumSize(QtCore.QSize(80, 55))
        self.frame_bug.setMaximumSize(QtCore.QSize(160, 55))
        self.frame_bug.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bug.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bug.setObjectName("frame_bug")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_bug)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.bn_VIPerson = QtWidgets.QPushButton(self.frame_bug)
        self.bn_VIPerson.setMinimumSize(QtCore.QSize(80, 55))
        self.bn_VIPerson.setMaximumSize(QtCore.QSize(160, 55))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.bn_VIPerson.setFont(font)
        self.bn_VIPerson.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border: none;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.bn_VIPerson.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("D:/Descargas/Minimalistic-Flat-Modern-GUI-Template-master/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/VisualImpare.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bn_VIPerson.setIcon(icon1)
        self.bn_VIPerson.setIconSize(QtCore.QSize(30, 40))
        self.bn_VIPerson.setFlat(True)
        self.bn_VIPerson.setObjectName("bn_VIPerson")
        self.horizontalLayout_16.addWidget(self.bn_VIPerson)
        self.verticalLayout_3.addWidget(self.frame_bug)
        self.frame_cloud = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_cloud.setMinimumSize(QtCore.QSize(80, 55))
        self.frame_cloud.setMaximumSize(QtCore.QSize(160, 55))
        self.frame_cloud.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_cloud.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_cloud.setObjectName("frame_cloud")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.frame_cloud)
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.bn_logItemRegister = QtWidgets.QPushButton(self.frame_cloud)
        self.bn_logItemRegister.setMinimumSize(QtCore.QSize(80, 55))
        self.bn_logItemRegister.setMaximumSize(QtCore.QSize(160, 55))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.bn_logItemRegister.setFont(font)
        self.bn_logItemRegister.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border: none;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.bn_logItemRegister.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/recordfile.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bn_logItemRegister.setIcon(icon2)
        self.bn_logItemRegister.setIconSize(QtCore.QSize(22, 22))
        self.bn_logItemRegister.setFlat(True)
        self.bn_logItemRegister.setObjectName("bn_logItemRegister")
        self.horizontalLayout_17.addWidget(self.bn_logItemRegister)
        self.verticalLayout_3.addWidget(self.frame_cloud)
        self.frame_android = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_android.setMinimumSize(QtCore.QSize(80, 55))
        self.frame_android.setMaximumSize(QtCore.QSize(160, 55))
        self.frame_android.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_android.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_android.setObjectName("frame_android")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.frame_android)
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_18.setSpacing(0)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.bn_bug = QtWidgets.QPushButton(self.frame_android)
        self.bn_bug.setMinimumSize(QtCore.QSize(80, 55))
        self.bn_bug.setMaximumSize(QtCore.QSize(160, 55))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.bn_bug.setFont(font)
        self.bn_bug.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border: none;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.bn_bug.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/bugAsset 47.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bn_bug.setIcon(icon3)
        self.bn_bug.setIconSize(QtCore.QSize(22, 22))
        self.bn_bug.setFlat(True)
        self.bn_bug.setObjectName("bn_bug")
        self.horizontalLayout_18.addWidget(self.bn_bug)
        self.verticalLayout_3.addWidget(self.frame_android)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3.addLayout(self.verticalLayout_4)
        self.horizontalLayout_2.addWidget(self.frame_bottom_west)
        self.frame_bottom_east = QtWidgets.QFrame(self.frame_bottom)
        self.frame_bottom_east.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom_east.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom_east.setObjectName("frame_bottom_east")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_bottom_east)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(self.frame_bottom_east)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame)
        self.stackedWidget.setObjectName("stackedWidget")
        self.Home_Page = QtWidgets.QWidget()
        self.Home_Page.setObjectName("Home_Page")
        self.frame_4 = QtWidgets.QFrame(self.Home_Page)
        self.frame_4.setGeometry(QtCore.QRect(0, 0, 1281, 691))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(1, 0, 1281, 441))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(40)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:white;")
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_7.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setStyleSheet("color:white;")
        self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.verticalLayout_7.addWidget(self.label)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame_4)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(300, 490, 701, 123))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_4.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("../Documentos Trabajo de Grado/TGDocument-20210119T231458Z-001/TGDocument/Portada/Logo1P.png"))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_19.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("../Documentos Trabajo de Grado/TGDocument-20210119T231458Z-001/TGDocument/Portada/Logo2_5.png"))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_19.addWidget(self.label_3)
        self.stackedWidget.addWidget(self.Home_Page)
        self.Rectificacion = QtWidgets.QWidget()
        self.Rectificacion.setObjectName("Rectificacion")
        self.frame_8 = QtWidgets.QFrame(self.Rectificacion)
        self.frame_8.setGeometry(QtCore.QRect(0, 0, 1281, 691))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.frame_8)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(10, 310, 1261, 321))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.groupBox_4 = QtWidgets.QGroupBox(self.horizontalLayoutWidget_4)
        self.groupBox_4.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_4.setObjectName("groupBox_4")
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox_4)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 20, 259, 287))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setStyleSheet("color:white;")
        self.label_6.setTextFormat(QtCore.Qt.PlainText)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setStyleSheet("color:white;")
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_8.setStyleSheet("color:white;")
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_9.setStyleSheet("color:white;")
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_10 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_10.setStyleSheet("color:white;")
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.label_11 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_11.setStyleSheet("color:white;")
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.label_12 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_12.setStyleSheet("color:white;")
        self.label_12.setObjectName("label_12")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.label_13 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_13.setStyleSheet("color:white;")
        self.label_13.setObjectName("label_13")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.label_14 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_14.setStyleSheet("color:white;")
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.label_15 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_15.setStyleSheet("color:white;")
        self.label_15.setObjectName("label_15")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.label_17 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_17.setStyleSheet("color:white;")
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.label_16 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_16.setStyleSheet("color:white;")
        self.label_16.setObjectName("label_16")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.Fx_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.Fx_left_camera.setObjectName("Fx_left_camera")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.Fx_left_camera)
        self.Fy_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.Fy_left_camera.setObjectName("Fy_left_camera")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.Fy_left_camera)
        self.Cx_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.Cx_left_camera.setObjectName("Cx_left_camera")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.Cx_left_camera)
        self.Cy_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.Cy_left_camera.setObjectName("Cy_left_camera")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.Cy_left_camera)
        self.K1_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.K1_left_camera.setObjectName("K1_left_camera")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.K1_left_camera)
        self.K2_left_camera_2 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.K2_left_camera_2.setObjectName("K2_left_camera_2")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.K2_left_camera_2)
        self.K3_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.K3_left_camera.setObjectName("K3_left_camera")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.K3_left_camera)
        self.P1_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.P1_left_camera.setObjectName("P1_left_camera")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.P1_left_camera)
        self.P2_left_camera = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.P2_left_camera.setObjectName("P2_left_camera")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.P2_left_camera)
        self.label_31 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_31.setStyleSheet("color:white;")
        self.label_31.setObjectName("label_31")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.label_31)
        self.horizontalLayout_20.addWidget(self.groupBox_4)
        self.groupBox_7 = QtWidgets.QGroupBox(self.horizontalLayoutWidget_4)
        self.groupBox_7.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_7.setObjectName("groupBox_7")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_7)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 259, 287))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_18 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_18.setStyleSheet("color:white;")
        self.label_18.setTextFormat(QtCore.Qt.PlainText)
        self.label_18.setObjectName("label_18")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.label_19 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_19.setStyleSheet("color:white;")
        self.label_19.setObjectName("label_19")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.Fx_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.Fx_right_camera.setObjectName("Fx_right_camera")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.Fx_right_camera)
        self.label_20 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_20.setStyleSheet("color:white;")
        self.label_20.setObjectName("label_20")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.Fy_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.Fy_right_camera.setObjectName("Fy_right_camera")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.Fy_right_camera)
        self.label_21 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_21.setStyleSheet("color:white;")
        self.label_21.setObjectName("label_21")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.label_22 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_22.setStyleSheet("color:white;")
        self.label_22.setObjectName("label_22")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.Cx_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.Cx_right_camera.setObjectName("Cx_right_camera")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.Cx_right_camera)
        self.label_23 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_23.setStyleSheet("color:white;")
        self.label_23.setObjectName("label_23")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_23)
        self.Cy_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.Cy_right_camera.setObjectName("Cy_right_camera")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.Cy_right_camera)
        self.label_24 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_24.setStyleSheet("color:white;")
        self.label_24.setObjectName("label_24")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.label_30 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_30.setStyleSheet("color:white;")
        self.label_30.setObjectName("label_30")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.label_30)
        self.label_25 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_25.setStyleSheet("color:white;")
        self.label_25.setObjectName("label_25")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_25)
        self.K1_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.K1_right_camera.setObjectName("K1_right_camera")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.K1_right_camera)
        self.label_26 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_26.setStyleSheet("color:white;")
        self.label_26.setObjectName("label_26")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.K2_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.K2_right_camera.setObjectName("K2_right_camera")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.K2_right_camera)
        self.label_27 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_27.setStyleSheet("color:white;")
        self.label_27.setObjectName("label_27")
        self.formLayout_2.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_27)
        self.K3_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.K3_right_camera.setObjectName("K3_right_camera")
        self.formLayout_2.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.K3_right_camera)
        self.label_29 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_29.setStyleSheet("color:white;")
        self.label_29.setObjectName("label_29")
        self.formLayout_2.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_29)
        self.P1_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.P1_right_camera.setObjectName("P1_right_camera")
        self.formLayout_2.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.P1_right_camera)
        self.label_28 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_28.setStyleSheet("color:white;")
        self.label_28.setObjectName("label_28")
        self.formLayout_2.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_28)
        self.P2_right_camera = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.P2_right_camera.setObjectName("P2_right_camera")
        self.formLayout_2.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.P2_right_camera)
        self.horizontalLayout_20.addWidget(self.groupBox_7)
        self.groupBox_8 = QtWidgets.QGroupBox(self.horizontalLayoutWidget_4)
        self.groupBox_8.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_8.setObjectName("groupBox_8")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_8)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(10, 20, 236, 209))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_33 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_33.setStyleSheet("color:white;")
        self.label_33.setObjectName("label_33")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_33)
        self.Baseline_depth = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.Baseline_depth.setObjectName("Baseline_depth")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.Baseline_depth)
        self.label_36 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_36.setStyleSheet("color:white;")
        self.label_36.setObjectName("label_36")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_36)
        self.label_37 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_37.setStyleSheet("color:white;")
        self.label_37.setObjectName("label_37")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_37)
        self.Ty_depth = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.Ty_depth.setObjectName("Ty_depth")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.Ty_depth)
        self.label_38 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_38.setStyleSheet("color:white;")
        self.label_38.setObjectName("label_38")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_38)
        self.Tz_depth = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.Tz_depth.setObjectName("Tz_depth")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.Tz_depth)
        self.label_39 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_39.setStyleSheet("color:white;")
        self.label_39.setObjectName("label_39")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_39)
        self.label_40 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_40.setStyleSheet("color:white;")
        self.label_40.setObjectName("label_40")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_40)
        self.Rx_depth = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.Rx_depth.setObjectName("Rx_depth")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.Rx_depth)
        self.label_41 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_41.setStyleSheet("color:white;")
        self.label_41.setObjectName("label_41")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_41)
        self.Rz_depth = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.Rz_depth.setObjectName("Rz_depth")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.Rz_depth)
        self.label_43 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_43.setStyleSheet("color:white;")
        self.label_43.setObjectName("label_43")
        self.formLayout_3.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_43)
        self.Cv_depth = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.Cv_depth.setObjectName("Cv_depth")
        self.formLayout_3.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.Cv_depth)
        self.label_32 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_32.setStyleSheet("color:white;")
        self.label_32.setObjectName("label_32")
        self.formLayout_3.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_32)
        self.formLayoutWidget_4 = QtWidgets.QWidget(self.groupBox_8)
        self.formLayoutWidget_4.setGeometry(QtCore.QRect(10, 250, 221, 46))
        self.formLayoutWidget_4.setObjectName("formLayoutWidget_4")
        self.formLayout_4 = QtWidgets.QFormLayout(self.formLayoutWidget_4)
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.formLayout_4.setObjectName("formLayout_4")
        self.Open_rectificationfile = QtWidgets.QLineEdit(self.formLayoutWidget_4)
        self.Open_rectificationfile.setObjectName("Open_rectificationfile")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.Open_rectificationfile)
        self.pushButton_2 = QtWidgets.QPushButton(self.formLayoutWidget_4)
        self.pushButton_2.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.pushButton_2.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("D:/Descargas/Minimalistic-Flat-Modern-GUI-Template-master/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/OpenFile.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon4)
        self.pushButton_2.setObjectName("pushButton_2")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.formLayoutWidget_4)
        self.pushButton_3.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/ForwardAsset.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon5)
        self.pushButton_3.setObjectName("pushButton_3")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_3)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.Preview = QtWidgets.QPushButton(self.formLayoutWidget_4)
        self.Preview.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Preview.setObjectName("Preview")
        self.horizontalLayout_24.addWidget(self.Preview)
        self.Rectification_2 = QtWidgets.QPushButton(self.formLayoutWidget_4)
        self.Rectification_2.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Rectification_2.setObjectName("Rectification_2")
        self.horizontalLayout_24.addWidget(self.Rectification_2)
        self.formLayout_4.setLayout(1, QtWidgets.QFormLayout.LabelRole, self.horizontalLayout_24)
        self.horizontalLayout_20.addWidget(self.groupBox_8)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.frame_8)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 1262, 301))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.left_camera = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.left_camera.setMinimumSize(QtCore.QSize(416, 299))
        self.left_camera.setStyleSheet("color:white;")
        self.left_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.left_camera.setObjectName("left_camera")
        self.horizontalLayout_3.addWidget(self.left_camera)
        self.right_camera = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.right_camera.setMinimumSize(QtCore.QSize(416, 299))
        self.right_camera.setStyleSheet("color:white;")
        self.right_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.right_camera.setObjectName("right_camera")
        self.horizontalLayout_3.addWidget(self.right_camera)
        self.disparity_map = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.disparity_map.setMinimumSize(QtCore.QSize(416, 299))
        self.disparity_map.setStyleSheet("color:white;")
        self.disparity_map.setAlignment(QtCore.Qt.AlignCenter)
        self.disparity_map.setObjectName("disparity_map")
        self.horizontalLayout_3.addWidget(self.disparity_map)
        self.stackedWidget.addWidget(self.Rectificacion)
        self.Local_Map = QtWidgets.QWidget()
        self.Local_Map.setObjectName("Local_Map")
        self.frame_2 = QtWidgets.QFrame(self.Local_Map)
        self.frame_2.setGeometry(QtCore.QRect(0, 0, 1281, 691))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.groupBox = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 1271, 681))
        self.groupBox.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 10, 1261, 671))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.frame_7 = QtWidgets.QFrame(self.horizontalLayoutWidget_3)
        self.frame_7.setStyleSheet("QFrame{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.Preview_camera = QtWidgets.QGraphicsView(self.frame_7)
        self.Preview_camera.setGeometry(QtCore.QRect(10, 10, 611, 451))
        self.Preview_camera.setObjectName("Preview_camera")
        self.groupBox_10 = QtWidgets.QGroupBox(self.frame_7)
        self.groupBox_10.setGeometry(QtCore.QRect(10, 480, 321, 121))
        self.groupBox_10.setObjectName("groupBox_10")
        self.formLayoutWidget_5 = QtWidgets.QWidget(self.groupBox_10)
        self.formLayoutWidget_5.setGeometry(QtCore.QRect(10, 20, 311, 93))
        self.formLayoutWidget_5.setObjectName("formLayoutWidget_5")
        self.formLayout_5 = QtWidgets.QFormLayout(self.formLayoutWidget_5)
        self.formLayout_5.setContentsMargins(0, 0, 0, 0)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_34 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_34.setStyleSheet("color:white;")
        self.label_34.setObjectName("label_34")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_34)
        self.depth_meter = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.depth_meter.setStyleSheet("color:white;")
        self.depth_meter.setText("")
        self.depth_meter.setObjectName("depth_meter")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.depth_meter)
        self.label_49 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_49.setStyleSheet("color:white;")
        self.label_49.setObjectName("label_49")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_49)
        self.label_50 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_50.setStyleSheet("color:white;")
        self.label_50.setObjectName("label_50")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_50)
        self.label_51 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_51.setStyleSheet("color:white;")
        self.label_51.setObjectName("label_51")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_51)
        self.Angle_Alpha = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.Angle_Alpha.setStyleSheet("color:white;")
        self.Angle_Alpha.setText("")
        self.Angle_Alpha.setObjectName("Angle_Alpha")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.Angle_Alpha)
        self.Depth_Angle_play = QtWidgets.QPushButton(self.formLayoutWidget_5)
        self.Depth_Angle_play.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/playasset.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Depth_Angle_play.setIcon(icon6)
        self.Depth_Angle_play.setObjectName("Depth_Angle_play")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.Depth_Angle_play)
        self.Angle_Theta = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.Angle_Theta.setStyleSheet("color:white;")
        self.Angle_Theta.setText("")
        self.Angle_Theta.setObjectName("Angle_Theta")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.Angle_Theta)
        self.horizontalLayout_22.addWidget(self.frame_7)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.groupBox_5 = QtWidgets.QGroupBox(self.horizontalLayoutWidget_3)
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_9.setGeometry(QtCore.QRect(10, 10, 331, 331))
        self.groupBox_9.setObjectName("groupBox_9")
        self.label_42 = QtWidgets.QLabel(self.groupBox_9)
        self.label_42.setGeometry(QtCore.QRect(10, 20, 151, 31))
        self.label_42.setStyleSheet("color:white;")
        self.label_42.setObjectName("label_42")
        self.InputDevice_play = QtWidgets.QPushButton(self.groupBox_9)
        self.InputDevice_play.setGeometry(QtCore.QRect(220, 200, 81, 31))
        self.InputDevice_play.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.InputDevice_play.setIcon(icon6)
        self.InputDevice_play.setObjectName("InputDevice_play")
        self.label_48 = QtWidgets.QLabel(self.groupBox_9)
        self.label_48.setGeometry(QtCore.QRect(250, 100, 71, 31))
        self.label_48.setStyleSheet("color:white;")
        self.label_48.setObjectName("label_48")
        self.label_46 = QtWidgets.QLabel(self.groupBox_9)
        self.label_46.setGeometry(QtCore.QRect(10, 170, 171, 31))
        self.label_46.setStyleSheet("color:white;")
        self.label_46.setObjectName("label_46")
        self.listOutputDevice = QtWidgets.QListView(self.groupBox_9)
        self.listOutputDevice.setGeometry(QtCore.QRect(10, 50, 191, 61))
        self.listOutputDevice.setObjectName("listOutputDevice")
        self.label_47 = QtWidgets.QLabel(self.groupBox_9)
        self.label_47.setGeometry(QtCore.QRect(210, 240, 101, 31))
        self.label_47.setStyleSheet("color:white;")
        self.label_47.setObjectName("label_47")
        self.listInputDevice = QtWidgets.QListView(self.groupBox_9)
        self.listInputDevice.setGeometry(QtCore.QRect(10, 200, 191, 61))
        self.listInputDevice.setObjectName("listInputDevice")
        self.OutputDevice_play = QtWidgets.QPushButton(self.groupBox_9)
        self.OutputDevice_play.setGeometry(QtCore.QRect(220, 50, 81, 31))
        self.OutputDevice_play.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.OutputDevice_play.setIcon(icon6)
        self.OutputDevice_play.setObjectName("OutputDevice_play")
        self.lcdNumberOutputLevel = QtWidgets.QLCDNumber(self.groupBox_9)
        self.lcdNumberOutputLevel.setGeometry(QtCore.QRect(230, 280, 61, 31))
        self.lcdNumberOutputLevel.setObjectName("lcdNumberOutputLevel")
        self.dialAudioLevel = QtWidgets.QDial(self.groupBox_9)
        self.dialAudioLevel.setGeometry(QtCore.QRect(200, 80, 50, 64))
        self.dialAudioLevel.setObjectName("dialAudioLevel")
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.groupBox_5)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(200, 350, 141, 31))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.Back_1 = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        self.Back_1.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("../../Documents/GUIUVVIS/Minimalistic-Flat-Modern-GUI-Template-master/icons/1x/backAsset.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Back_1.setIcon(icon7)
        self.Back_1.setObjectName("Back_1")
        self.horizontalLayout_21.addWidget(self.Back_1)
        self.Next_1 = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        self.Next_1.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Next_1.setIcon(icon5)
        self.Next_1.setObjectName("Next_1")
        self.horizontalLayout_21.addWidget(self.Next_1)
        self.verticalLayout_6.addWidget(self.groupBox_5)
        self.horizontalLayout_22.addLayout(self.verticalLayout_6)
        self.stackedWidget.addWidget(self.Local_Map)
        self.Tutorial = QtWidgets.QWidget()
        self.Tutorial.setObjectName("Tutorial")
        self.frame_6 = QtWidgets.QFrame(self.Tutorial)
        self.frame_6.setGeometry(QtCore.QRect(0, 0, 1281, 701))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.Tutorial_graph = QtWidgets.QGroupBox(self.frame_6)
        self.Tutorial_graph.setGeometry(QtCore.QRect(10, 10, 1261, 521))
        self.Tutorial_graph.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.Tutorial_graph.setObjectName("Tutorial_graph")
        self.groupBox_13 = QtWidgets.QGroupBox(self.frame_6)
        self.groupBox_13.setGeometry(QtCore.QRect(10, 560, 221, 51))
        self.groupBox_13.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_13.setObjectName("groupBox_13")
        self.formLayoutWidget_6 = QtWidgets.QWidget(self.groupBox_13)
        self.formLayoutWidget_6.setGeometry(QtCore.QRect(9, 19, 135, 21))
        self.formLayoutWidget_6.setObjectName("formLayoutWidget_6")
        self.formLayout_6 = QtWidgets.QFormLayout(self.formLayoutWidget_6)
        self.formLayout_6.setContentsMargins(0, 0, 0, 0)
        self.formLayout_6.setObjectName("formLayout_6")
        self.Tuto_Back = QtWidgets.QPushButton(self.formLayoutWidget_6)
        self.Tuto_Back.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Tuto_Back.setIcon(icon7)
        self.Tuto_Back.setObjectName("Tuto_Back")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.Tuto_Back)
        self.Tuto_Next = QtWidgets.QPushButton(self.formLayoutWidget_6)
        self.Tuto_Next.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Tuto_Next.setIcon(icon5)
        self.Tuto_Next.setObjectName("Tuto_Next")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.Tuto_Next)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.frame_6)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(1090, 570, 174, 31))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.Back_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        self.Back_2.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Back_2.setIcon(icon7)
        self.Back_2.setObjectName("Back_2")
        self.horizontalLayout_23.addWidget(self.Back_2)
        self.Next_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        self.Next_2.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Next_2.setIcon(icon6)
        self.Next_2.setObjectName("Next_2")
        self.horizontalLayout_23.addWidget(self.Next_2)
        self.stackedWidget.addWidget(self.Tutorial)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.frame_9 = QtWidgets.QFrame(self.page)
        self.frame_9.setGeometry(QtCore.QRect(0, 0, 1281, 691))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.groupBox_14 = QtWidgets.QGroupBox(self.frame_9)
        self.groupBox_14.setGeometry(QtCore.QRect(20, 10, 1260, 610))
        self.groupBox_14.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_14.setObjectName("groupBox_14")
        self.groupBox_15 = QtWidgets.QGroupBox(self.groupBox_14)
        self.groupBox_15.setGeometry(QtCore.QRect(10, 20, 1240, 281))
        self.groupBox_15.setObjectName("groupBox_15")
        self.Image_DetectObject = QtWidgets.QGraphicsView(self.groupBox_15)
        self.Image_DetectObject.setGeometry(QtCore.QRect(10, 20, 1220, 241))
        self.Image_DetectObject.setObjectName("Image_DetectObject")
        self.groupBox_16 = QtWidgets.QGroupBox(self.groupBox_14)
        self.groupBox_16.setGeometry(QtCore.QRect(10, 310, 1240, 280))
        self.groupBox_16.setObjectName("groupBox_16")
        self.Image_DepthMap = QtWidgets.QGraphicsView(self.groupBox_16)
        self.Image_DepthMap.setGeometry(QtCore.QRect(10, 20, 1220, 240))
        self.Image_DepthMap.setObjectName("Image_DepthMap")
        self.stackedWidget.addWidget(self.page)
        self.File_Records = QtWidgets.QWidget()
        self.File_Records.setObjectName("File_Records")
        self.frame_3 = QtWidgets.QFrame(self.File_Records)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 891, 601))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 251, 161))
        self.groupBox_2.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_2.setObjectName("groupBox_2")
        self.Accept = QtWidgets.QPushButton(self.groupBox_2)
        self.Accept.setGeometry(QtCore.QRect(180, 120, 61, 23))
        self.Accept.setStyleSheet("QPushButton {\n"
"    color:white;\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    background-color: rgba(0,0,0,0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgb(91,90,90);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: rgba(0,0,0,0);\n"
"}")
        self.Accept.setObjectName("Accept")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 231, 91))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.ObjectReconice_menu = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.ObjectReconice_menu.setContentsMargins(0, 0, 0, 0)
        self.ObjectReconice_menu.setObjectName("ObjectReconice_menu")
        self.radioButton_1 = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioButton_1.setStyleSheet("QRadioButton {\n"
"    background:rgb(91,90,90);\n"
"    color:white;\n"
"}\n"
"QRadioButton::indicator {\n"
"    width:10px;\n"
"    height:10px;\n"
"    border-radius: 7px;\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background-color:rgb(0,143,170);\n"
"    border: 2px solid rgb(51,51,51);\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked {\n"
"    background-color:rgb(91,90,90);\n"
"    border:2px solid rgb(51,51,51);\n"
"}")
        self.radioButton_1.setObjectName("radioButton_1")
        self.ObjectReconice_menu.addWidget(self.radioButton_1)
        self.radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioButton.setStyleSheet("QRadioButton {\n"
"    background:rgb(91,90,90);\n"
"    color:white;\n"
"}\n"
"QRadioButton::indicator {\n"
"    width:10px;\n"
"    height:10px;\n"
"    border-radius: 7px;\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background-color:rgb(0,143,170);\n"
"    border: 2px solid rgb(51,51,51);\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked {\n"
"    background-color:rgb(91,90,90);\n"
"    border:2px solid rgb(51,51,51);\n"
"}")
        self.radioButton.setObjectName("radioButton")
        self.ObjectReconice_menu.addWidget(self.radioButton)
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox_3.setGeometry(QtCore.QRect(270, 10, 611, 581))
        self.groupBox_3.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_3.setObjectName("groupBox_3")
        self.ObjectReconice_log = QtWidgets.QTextBrowser(self.groupBox_3)
        self.ObjectReconice_log.setGeometry(QtCore.QRect(10, 20, 591, 551))
        self.ObjectReconice_log.setObjectName("ObjectReconice_log")
        self.groupBox_6 = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox_6.setGeometry(QtCore.QRect(10, 180, 251, 411))
        self.groupBox_6.setStyleSheet("QGroupBox{\n"
"    border:1px solid rgb(51,51,51);    \n"
"    border-radius:4px;\n"
"    color:white;\n"
"    background:rgb(91,90,90);\n"
"}")
        self.groupBox_6.setObjectName("groupBox_6")
        self.AdicionalText = QtWidgets.QTextBrowser(self.groupBox_6)
        self.AdicionalText.setGeometry(QtCore.QRect(10, 20, 231, 381))
        self.AdicionalText.setObjectName("AdicionalText")
        self.stackedWidget.addWidget(self.File_Records)
        self.Error_report = QtWidgets.QWidget()
        self.Error_report.setObjectName("Error_report")
        self.frame_5 = QtWidgets.QFrame(self.Error_report)
        self.frame_5.setGeometry(QtCore.QRect(0, 0, 1271, 691))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.Error_Log = QtWidgets.QTextBrowser(self.frame_5)
        self.Error_Log.setGeometry(QtCore.QRect(10, 10, 1251, 671))
        self.Error_Log.setObjectName("Error_Log")
        self.stackedWidget.addWidget(self.Error_report)
        self.horizontalLayout_14.addWidget(self.stackedWidget)
        self.verticalLayout_2.addWidget(self.frame)
        self.frame_low = QtWidgets.QFrame(self.frame_bottom_east)
        self.frame_low.setMinimumSize(QtCore.QSize(0, 20))
        self.frame_low.setMaximumSize(QtCore.QSize(16777215, 20))
        self.frame_low.setStyleSheet("")
        self.frame_low.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_low.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_low.setObjectName("frame_low")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.frame_low)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.frame_tab = QtWidgets.QFrame(self.frame_low)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.frame_tab.setFont(font)
        self.frame_tab.setStyleSheet("background:rgb(51,51,51);")
        self.frame_tab.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_tab.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_tab.setObjectName("frame_tab")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.frame_tab)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.lab_tab = QtWidgets.QLabel(self.frame_tab)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Light")
        font.setPointSize(10)
        self.lab_tab.setFont(font)
        self.lab_tab.setStyleSheet("color:rgb(255,255,255);")
        self.lab_tab.setObjectName("lab_tab")
        self.horizontalLayout_12.addWidget(self.lab_tab)
        self.horizontalLayout_11.addWidget(self.frame_tab)
        self.frame_drag = QtWidgets.QFrame(self.frame_low)
        self.frame_drag.setMinimumSize(QtCore.QSize(20, 20))
        self.frame_drag.setMaximumSize(QtCore.QSize(20, 20))
        self.frame_drag.setStyleSheet("background:rgb(51,51,51);")
        self.frame_drag.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_drag.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_drag.setObjectName("frame_drag")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.frame_drag)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.horizontalLayout_11.addWidget(self.frame_drag)
        self.verticalLayout_2.addWidget(self.frame_low)
        self.horizontalLayout_2.addWidget(self.frame_bottom_east)
        self.verticalLayout.addWidget(self.frame_bottom)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.bn_home.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.bn_VIPerson.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.bn_logItemRegister.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(5))
        self.bn_bug.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        #Callbacks button into VIPerson configuration GUI
        self.pushButton_3.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.Next_1.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        self.Next_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
        self.Back_1.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.Back_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
         # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.Preview.clicked.connect(self.thread.change_pixmap_signal.connect(self.update_image))
        # start the thread
        self.thread.start()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.left_camera.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "UVVIS-Interfaz de Usuario Supervisor"))
        self.lab_appname.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.lab_user.setText(_translate("MainWindow", "<html><head/><body><p>Usuario Supervisor</p></body></html>"))
        self.bn_home.setToolTip(_translate("MainWindow", "Home"))
        self.bn_VIPerson.setToolTip(_translate("MainWindow", "Android"))
        self.bn_logItemRegister.setToolTip(_translate("MainWindow", "Cloud"))
        self.bn_bug.setToolTip(_translate("MainWindow", "Bug"))
        self.label_2.setText(_translate("MainWindow", "Bienvenido a la GUI\n"
"UVVIS"))
        self.label.setText(_translate("MainWindow", "El sistema se encuentra en el modo de usuario supervisor por tanto la herramienta presenta opciones\n"
"para que el usuario pueda acceder al mapa local, los registros de objetos reconocidos, errores del\n"
"sistema y configurar la herramienta para operar el modo de usuario con impedimento visual a trav??s\n"
"todo a trav??s de esta interfaz gr??fica de usuario."))
        self.groupBox_4.setTitle(_translate("MainWindow", "Par??metros c??mara est??reo izquierda"))
        self.label_6.setText(_translate("MainWindow", "Distancia focal (px)"))
        self.label_7.setText(_translate("MainWindow", "Fx"))
        self.label_8.setText(_translate("MainWindow", "Fy"))
        self.label_9.setText(_translate("MainWindow", "Posici??n central (px)"))
        self.label_10.setText(_translate("MainWindow", "Cx"))
        self.label_11.setText(_translate("MainWindow", "Cy"))
        self.label_12.setText(_translate("MainWindow", "Coeficiente de distorsi??n"))
        self.label_13.setText(_translate("MainWindow", "K1"))
        self.label_14.setText(_translate("MainWindow", "K2"))
        self.label_15.setText(_translate("MainWindow", "K3"))
        self.label_17.setText(_translate("MainWindow", "P2"))
        self.label_16.setText(_translate("MainWindow", "P1"))
        self.label_31.setText(_translate("MainWindow", "K-Radial, P-Tangencial"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Par??metros c??mara est??reo derecha"))
        self.label_18.setText(_translate("MainWindow", "Distancia focal (px)"))
        self.label_19.setText(_translate("MainWindow", "Fx"))
        self.label_20.setText(_translate("MainWindow", "Fy"))
        self.label_21.setText(_translate("MainWindow", "Posici??n central (px)"))
        self.label_22.setText(_translate("MainWindow", "Cx"))
        self.label_23.setText(_translate("MainWindow", "Cy"))
        self.label_24.setText(_translate("MainWindow", "Coeficiente de distorsi??n"))
        self.label_30.setText(_translate("MainWindow", "K-Radial, P-Tangencial"))
        self.label_25.setText(_translate("MainWindow", "K1"))
        self.label_26.setText(_translate("MainWindow", "K2"))
        self.label_27.setText(_translate("MainWindow", "K3"))
        self.label_29.setText(_translate("MainWindow", "P1"))
        self.label_28.setText(_translate("MainWindow", "P2"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Par??metros de calibraci??n c??mara est??reo"))
        self.label_33.setText(_translate("MainWindow", "Baseline"))
        self.label_36.setText(_translate("MainWindow", "Vector de traslaci??n"))
        self.label_37.setText(_translate("MainWindow", "Ty"))
        self.label_38.setText(_translate("MainWindow", "Tz"))
        self.label_39.setText(_translate("MainWindow", "Vector de rotaci??n"))
        self.label_40.setText(_translate("MainWindow", "Rx"))
        self.label_41.setText(_translate("MainWindow", "Rz"))
        self.label_43.setText(_translate("MainWindow", "Cv"))
        self.label_32.setText(_translate("MainWindow", "Bias"))
        self.pushButton_3.setText(_translate("MainWindow", "Siguiente"))
        self.Preview.setText(_translate("MainWindow", "Previsualizaci??n"))
        self.Rectification_2.setText(_translate("MainWindow", "Rectificaci??n"))
        self.left_camera.setText(_translate("MainWindow", "Imagen C??mara Izquierda"))
        self.right_camera.setText(_translate("MainWindow", "Imagen C??mara Derecha"))
        self.disparity_map.setText(_translate("MainWindow", "Mapa de disparidad"))
        self.groupBox_10.setTitle(_translate("MainWindow", "Prueba de funcionamiento de orientaci??n y profundidad  "))
        self.label_34.setText(_translate("MainWindow", "Profundidad (m):"))
        self.label_49.setText(_translate("MainWindow", "??ngulo (Alpha)"))
        self.label_50.setText(_translate("MainWindow", "??ngulo (Theta)"))
        self.label_51.setText(_translate("MainWindow", "Clickear en el frame en la parte\n"
"superior antes de presionar el bot??n"))
        self.Depth_Angle_play.setText(_translate("MainWindow", "Reproducir\n"
"prueba"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Configuraci??n de dispositivos de sonido"))
        self.label_42.setText(_translate("MainWindow", "Seleccionar dispositivo de salida"))
        self.InputDevice_play.setText(_translate("MainWindow", "Reproducir\n"
"prueba"))
        self.label_48.setText(_translate("MainWindow", "Nivel de Audio\n"
"de Salida"))
        self.label_46.setText(_translate("MainWindow", "Seleccionar dispositivo de entrada"))
        self.label_47.setText(_translate("MainWindow", "Prueba de dispositivo\n"
"entrada"))
        self.OutputDevice_play.setText(_translate("MainWindow", "Reproducir\n"
"prueba"))
        self.Back_1.setText(_translate("MainWindow", "Anterior"))
        self.Next_1.setText(_translate("MainWindow", "Siguiente"))
        self.Tutorial_graph.setTitle(_translate("MainWindow", "Tutorial de uso de hardware en modo de usuario con discapacidad visual "))
        self.groupBox_13.setTitle(_translate("MainWindow", "Men?? de navegaci??n de tutorial"))
        self.Tuto_Back.setText(_translate("MainWindow", "Anterior"))
        self.Tuto_Next.setText(_translate("MainWindow", "Siguiente"))
        self.Back_2.setText(_translate("MainWindow", "Anterior"))
        self.Next_2.setText(_translate("MainWindow", "Empezar"))
        self.groupBox_14.setTitle(_translate("MainWindow", "Visualizaci??n de imagen de c??mara con objetos detectados y mapa de profundidad"))
        self.groupBox_15.setTitle(_translate("MainWindow", "Imagen con objetos detectados"))
        self.groupBox_16.setTitle(_translate("MainWindow", "Mapa de profundidad"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Men?? de Registro de Objetos Reconocidos"))
        self.Accept.setText(_translate("MainWindow", "Aplicar"))
        self.radioButton_1.setText(_translate("MainWindow", "Abrir Registro de Objetos Reconocidos "))
        self.radioButton.setText(_translate("MainWindow", "Borrar Registro de Objetos Reconocidos"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Visualizaci??n de Registro de Objetos Reconocidos"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Informaci??n Adicional"))
        self.lab_tab.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.frame_drag.setToolTip(_translate("MainWindow", "Drag"))

class VideoThread(QThread, Ui_MainWindow):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

