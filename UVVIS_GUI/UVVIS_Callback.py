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
from gtts import gTTS
import pygame
from io import BytesIO
import math

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
        self.pushButton_3.clicked.connect(lambda: self.Next_Config())
        self.Back_1.clicked.connect(lambda: self.ActivateOtherTab())
        self.pushButton_2.clicked.connect(lambda: self.open_dialog_box())
        self.Rectification_2.clicked.connect(lambda: self.RectificactionCamera())
        self.Disparity_Map_Bt.clicked.connect(lambda: self.showmapadisparidad())
        self.Depth_Angle_play.clicked.connect(lambda: self.Distance_SoundPrueba())
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
        self.Disparity_Map_Bt.setEnabled(True)
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
            config.ViewActivate=1
            self.ShowPreviewMap.ImageUpdate.connect(self.ImageUpdatePreview)

    def ActivateOtherTab(self):
        config.ViewActivate=0
    
    def disparityList(self, Disparity_list):
        print(Disparity_list)
        if (Disparity_list[5] > 0):
            depth = (Disparity_list[0]/1.6) * (-Disparity_list[2] / Disparity_list[5])
            changeInX = Disparity_list[3] - Disparity_list[6][0]
            changeInY = Disparity_list[4] - Disparity_list[6][1]
            theta_angle= np.degrees(math.atan2(changeInY,changeInX))
        else:
            depth = 0
            theta_angle=0
        print("Profundidad " + '{0:.2f}'.format(depth / 1000) + " m"+" Angulo delta: "+'{0:1d}'.format(int(theta_angle)))
        gtts=gTTS (text = "Profundidad " + '{0:.2f}'.format(depth / 1000) + "m"+" Angulo delta: "+'{0:1d}'.format(int(theta_angle)), lang='es', slow=False)
        #self.textTovoice(gtts)
        config.ViewActivate=1

    def Distance_SoundPrueba(self):
        config.ViewActivate=2
        self.ShowDepthMap.disparityLog.connect(self.disparityList)
        

    def textTovoice(self,tts) :
        # convert to file-like object
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            #--- play it ---
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()