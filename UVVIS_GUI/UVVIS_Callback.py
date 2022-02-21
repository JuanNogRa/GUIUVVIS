# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:47:11 2021

@author: juano
"""
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QFileDialog
from UVVIS_Thread import *
from UVVIS_GUI import *
import config
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
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
        
        self.Preview.clicked.connect(lambda: self.ActivateCamera())
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.clicked.connect(lambda: self.Next_Config())
        self.Back_1.clicked.connect(lambda: self.Activate1())
        self.pushButton_2.clicked.connect(lambda: self.open_dialog_box())
        self.Rectification_2.clicked.connect(lambda: self.RectificactionCamera())
        self.Disparity_Map_Bt.clicked.connect(lambda: self.showmapadisparidad())
        self.ShowImageOnInterface = ShowImageOnInterface(" ", False)
        self.ShowPreviewMap = ShowPreviewMap(" ", False)
        self.ShowDepthMap = ShowDepthMap()
        self.Preview_camera.mousePressEvent = self.CalculateDepth
        
    def CalculateDepth(self, event):
        config.x = event.pos().x()
        config.y = event.pos().y()   
        
    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        self.path = filename[0]
        self.Open_rectificationfile.setText(self.path)
    
    def ImageUpdateLeftRect(self, Image):
        self.left_rect=Image
        
    def ImageUpdateRightRect(self, Image):
        self.right_rect=Image
        
    def ActivateCamera(self):
        self.ActivateRectification=False
        self.ShowImageOnInterface = ShowImageOnInterface("", self.ActivateRectification)
        if self.ShowImageOnInterface.isFinished:
            self.ShowImageOnInterface.start()
            self.ShowImageOnInterface.ImageUpdate.connect(self.ImageUpdateSlot)
            self.ShowImageOnInterface.ImageUpdate1.connect(self.ImageUpdateSlot1)
        
    def RectificactionCamera(self):
        self.ActivateRectification=True
        self.ShowImageOnInterface = ShowImageOnInterface(self.path,self.ActivateRectification)
        if self.ShowImageOnInterface.isFinished:
            self.ShowImageOnInterface.start()       
            self.ShowImageOnInterface.ImageUpdate.connect(self.ImageUpdateSlot)
            self.ShowImageOnInterface.ImageUpdate1.connect(self.ImageUpdateSlot1)   
    
    def showmapadisparidad(self):
        if self.ShowDepthMap.isFinished:
            self.ShowDepthMap.start()
            self.ShowDepthMap.ImageUpdate.connect(self.ImageUpdateSlotDepth)
            self.pushButton_3.setEnabled(True)
                
    def ImageUpdateSlot(self, Image):
        self.left_camera.setPixmap(QPixmap.fromImage(Image))
        
    def ImageUpdateSlot1(self, Image):
        self.right_camera.setPixmap(QPixmap.fromImage(Image))
                
    def ImageUpdateSlotDepth(self, Image):
        self.disparity_map.setPixmap(QPixmap.fromImage(Image))
    
    def ImageUpdatePreview(self, Image):
        self.Preview_camera.setPixmap(QPixmap.fromImage(Image))
    
    def Next_Config(self): 
        self.ShowPreviewMap = ShowPreviewMap(self.path,self.ActivateRectification)
        if self.ShowPreviewMap.isFinished:
            self.ShowPreviewMap.start()
            config.ViewActivate=True
            self.ShowPreviewMap.ImageUpdate.connect(self.ImageUpdatePreview)
    
    def Activate1(self):
        config.ViewActivate=False
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()