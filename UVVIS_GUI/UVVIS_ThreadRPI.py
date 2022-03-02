# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:41:22 2021

@author: juano
"""
from asyncio.windows_events import NULL
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSignal,QThread, Qt, QMutex 
import configparser
import cv2
import config
import torch
import numpy as np
from tflite_runtime.interpreter import Interpreter
import tensorflow as tf
import cv2

left_rect=np.zeros((1,1,1), np.uint8) 
right_rect=np.zeros((1,1,1), np.uint8)
scale_x=0
scale_y=0
mutex = QMutex()
Disparity=[]
class Resolution :
    width = 1280
    height = 720
#Hilo para imprimir en interfaz las imagenes en raw y rectificadas 
class ShowImageOnInterface(QThread):
    ImageUpdate = pyqtSignal(QImage)
    ImageUpdate1 = pyqtSignal(QImage)
    ParamsLabels = pyqtSignal(list)
    def __init__(self, path, activateRectification):
        QThread.__init__(self)
        self.path = path
        self.activateRectification=activateRectification
        
        
    def run(self):
        global left_rect, right_rect
        self.ThreadActive = True
        cap = cv2.VideoCapture(1,cv2.CAP_MSMF)
        image_size = Resolution()
        image_size.width = 1280
        image_size.height = 720
        # Set the video resolution to HD720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width*2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
        
        if self.path!="":
            camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y, left_param, right_param, camera_parameter = self.init_calibration(self.path, image_size)
            self.ParamsLabels.emit([left_param,right_param,camera_parameter])
        while self.ThreadActive:
            mutex.lock()
            ret, frame = cap.read()
            if ret:
                # Extract left and right images from side-by-side
                left_right_image = np.split(frame, 2, axis=1)
                #print(str(left_right_image[1].shape))
                if self.path!="" and self.activateRectification==True:
                    left_rect = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                    right_rect = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                    Image = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
                    Image1 = cv2.cvtColor(right_rect, cv2.COLOR_BGR2RGB)
                    left_rect = cv2.resize(Image, (800,600), interpolation= cv2.INTER_LINEAR)
                    right_rect = cv2.resize(Image1, (800,600), interpolation= cv2.INTER_LINEAR)
                    
                else:
                    Image = cv2.cvtColor(left_right_image[0], cv2.COLOR_BGR2RGB)
                    Image1 = cv2.cvtColor(left_right_image[1], cv2.COLOR_BGR2RGB)
                if config.ViewActivate==0:
                    convertToQtformat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)   
                    convertToQtformat1 = QImage(Image1.data, Image1.shape[1], Image1.shape[0], QImage.Format_RGB888)
                    Pic = convertToQtformat.scaled(250, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    Pic1 = convertToQtformat1.scaled(250, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.ImageUpdate.emit(Pic)
                    self.ImageUpdate1.emit(Pic1)
            mutex.unlock()
                
    def stop(self):
        self.ThreadActive = False
        self.quit()
        
    def init_calibration(self, calibration_file, image_size) :
        global Disparity
        cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])
    
        config = configparser.ConfigParser()
        config.read(calibration_file)
    
        check_data = True
        resolution_str = ''
        if image_size.width == 2208 :
            resolution_str = '2K'
        elif image_size.width == 1920 :
            resolution_str = 'FHD'
        elif image_size.width == 1280 :
            resolution_str = 'HD'
        elif image_size.width == 672 :
            resolution_str = 'VGA'
        else:
            resolution_str = 'HD'
            check_data = False
        
        T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                       float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                       float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])
    
        left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)
        left_param = [left_cam_cx, left_cam_cy, left_cam_fx, left_cam_fy, left_cam_k1, left_cam_k2, left_cam_p1, left_cam_p2
        ,left_cam_p3, left_cam_k3]
    
        right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_param = [right_cam_cx, right_cam_cy, right_cam_fx, right_cam_fy, right_cam_k1, right_cam_k2, right_cam_p1, right_cam_p2
        ,right_cam_p3, right_cam_k3]

        R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                          float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                          float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])
        camera_parameter=[T_,R_zed]
        R, _ = cv2.Rodrigues(R_zed)
        cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                             [0, left_cam_fy, left_cam_cy],
                             [0, 0, 1]])
    
        cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                              [0, right_cam_fy, right_cam_cy],
                              [0, 0, 1]])
    
        distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])
    
        distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])
    
        T = np.array([[T_[0]], [T_[1]], [T_[2]]])
        R1 = R2 = P1 = P2 = np.array([])
    
        R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                           cameraMatrix2=cameraMatrix_right,
                                           distCoeffs1=distCoeffs_left,
                                           distCoeffs2=distCoeffs_right,
                                           R=R, T=T,
                                           flags=cv2.CALIB_ZERO_DISPARITY,
                                           alpha=0,
                                           imageSize=(image_size.width, image_size.height),
                                           newImageSize=(image_size.width, image_size.height))[0:4]
    
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)
    
        cameraMatrix_left = P1
        cameraMatrix_right = P2

        Disparity=[left_cam_fx, left_cam_fy, T_[0], left_cam_cx, left_cam_cy]
        Disparity.append([0,0])
        Disparity.append([0,0])
        return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y, left_param, right_param, camera_parameter

class ShowDepthMap(QThread):
    global left_rect, right_rect, scale_x, scale_y, Disparity
    ImageUpdate = pyqtSignal(QImage)
    disparityLog = pyqtSignal(list)
    def __init__(self):
        QThread.__init__(self)
        
    def run(self):
        max_disparity = 128
        wls_lmbda = 800
        wls_sigma = 1.2
        
        self.ThreadActive = True
        
        while self.ThreadActive:
            mutex.lock()
            stereoSGBM = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities = max_disparity, # max_disp has to be dividable by 16 f. E. HH 192, 256
#             blockSize=window_size,
#             P1=8 * window_size ** 2,       # 8*number_of_image_channels*SADWindowSize*SADWindowSize
#             P2=32 * window_size ** 2,      # 32*number_of_image_channels*SADWindowSize*SADWindowSize
#             disp12MaxDiff=1,
#             uniquenessRatio=15,
#             speckleWindowSize=0,
#             speckleRange=2,
#             preFilterCap=63,
#             mode=cv2.STEREO_SGBM_MODE_HH
            )
            
        
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereoSGBM)
            wls_filter.setLambda(wls_lmbda)
            wls_filter.setSigmaColor(wls_sigma)

            # remember to convert to grayscale (as the disparity matching works on grayscale)

            grayL = cv2.cvtColor(left_rect,cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right_rect,cv2.COLOR_BGR2GRAY)
        
            # perform preprocessing - raise to the power, as this subjectively appears
            # to improve subsequent disparity calculation

            grayL = np.power(grayL, 0.75).astype('uint8')
            grayR = np.power(grayR, 0.75).astype('uint8')
        
            left_matcher=stereoSGBM
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        
            displ = left_matcher.compute(cv2.UMat(grayL),cv2.UMat(grayR))  # .astype(np.float32)/16
            dispr = right_matcher.compute(cv2.UMat(grayR),cv2.UMat(grayL))  # .astype(np.float32)/16
            displ = np.int16(cv2.UMat.get(displ))
            dispr = np.int16(cv2.UMat.get(dispr))
            disparity = wls_filter.filter(displ, grayL, None, dispr)
            
            # scale the disparity to 8-bit for viewing
            # divide by 16 and convert to 8-bit image (then range of values should
            # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
            # so we fix this also using a initial threshold between 0 and max_disparity
            # as disparity=-1 means no disparity available
            _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
            disparity_scaled = (disparity / 16.).astype(np.uint8) 
            Image = cv2.applyColorMap((disparity_scaled * (256. / max_disparity)).astype(np.uint8), cv2.COLORMAP_HOT)
            #Image = cv2.cvtColor(disparity_scaled, cv2.COLOR_GRAY2BGR)            
            if config.ViewActivate==0:
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888) 
                PicDepth = ConvertToQtFormat.scaled(250, 200, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(PicDepth)
            elif config.ViewActivate==2:
                Disparity[5]=(disparity_scaled[scale_y, scale_x])
                Disparity[6]=([scale_x, scale_y])
                self.disparityLog.emit(Disparity)
            elif config.ViewActivate==4:
                Disparity[5]=(disparity_scaled[scale_y, scale_x])
                Disparity[6]=([scale_x, scale_y])
                self.disparityLog.emit(Disparity)
            mutex.unlock()
            
    def stop(self):
        self.ThreadActive = False
        self.quit()  

class ShowPreviewMap(QThread):
    ImageUpdate = pyqtSignal(QImage)
    ImageUpdate1 = pyqtSignal(QImage)
    
    def __init__(self, path, activateRectification):
        QThread.__init__(self)
        self.path = path
        self.activateRectification=activateRectification
        
    def run(self):
        self.ThreadActive = True
        global scale_x, scale_y
        while self.ThreadActive:
            mutex.lock()
            if self.path!="" and self.activateRectification==True:
                if config.ViewActivate==1:
                    Image = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
                    scale_x = int(config.x*(Image.shape[1]/360))
                    scale_y = int(config.y*(Image.shape[0]/320))
                    Image = cv2.circle(Image, (scale_x,scale_y), 1, (0,255,255), 3)
                    convertToQtformat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)   
                    Pic = convertToQtformat.scaled(360, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.ImageUpdate.emit(Pic)
            else:
                print(str(self.activateRectification))                    
            mutex.unlock()
                
    def stop(self):
        self.ThreadActive = False
        self.quit()


class ShowInferenceModel(QThread):
    ImageUpdate = pyqtSignal(QImage)
    ObjectsDetect = pyqtSignal(list)
    global scale_x, scale_y
    def __init__(self, path, activateRectification):
        """
        Initializes the class with youtube url and output file.
        """
        QThread.__init__(self)
        self.path = path
        self.activateRectification=activateRectification
        self.interpreter = self.load_model()

    def run(self):
        self.ThreadActive = True
        #get input and output tensors
        # Get model details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_std = 255
        while self.ThreadActive:
            mutex.lock()
            image=left_rect
            frame = cv2.resize(left_rect, (320,320), interpolation= cv2.INTER_LINEAR)
            input_data = np.expand_dims(frame, 0)
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data)) / input_std
            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(input_details[0]['index'],input_data)
            self.interpreter.invoke()

            """Output data"""
            output_data = self.interpreter.get_tensor(output_details[0]['index'])  # get tensor  x(1, 25200, 7)
            xyxy, classes, scores, selected_indices = self.YOLOdetect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]
            selected_indices=selected_indices.numpy()
            nms_classes=[]
            nms_xyxy=[]
            nms_score=[]
            for i in range(len(selected_indices)):
                nms_classes.append(classes[selected_indices[i]])
                nms_score.append(scores[selected_indices[i]])
                nms_xyxy.append([xyxy[0][selected_indices[i]], xyxy[1][selected_indices[i]], xyxy[2][selected_indices[i]], xyxy[3][selected_indices[i]]])
            classes=nms_classes
            xyxy=nms_xyxy
            scores=nms_score
            labels = ['Armario', 'Bañera', 'Bote de Basura', 'Cama', 'Ducha', 'Escaleras', 'Lavamanos', 'Lavaplatos', 'Sanitario']
            image=self.plot_boxes(scores, xyxy, image, classes,labels)
            if config.ViewActivate==3:
                convertToQtformat = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)   
                Pic = convertToQtformat.scaled(360, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ImageUpdate.emit(Pic)
            mutex.unlock()

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # Load TFLite model and allocate tensors.
        interpreter = Interpreter(model_path='bestPack2-fp16.tflite')
        #allocate the tensors
        interpreter.allocate_tensors()
        return interpreter

    def classFilter(self,classdata):
        classes = []  # create a list
        for i in range(classdata.shape[0]):         # loop through all predictions
            classes.append(classdata[i].argmax())   # get the best classification location
        return classes  # return classes (int)

    def YOLOdetect(self,output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
        output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
        boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
        scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
        classes = self.classFilter(output_data[..., 5:]) # get classes
        selected_indices = tf.image.non_max_suppression(
        boxes, scores, 1, iou_threshold=0.5,
        score_threshold=float('-inf'), name=None
        )
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

        return xyxy, classes, scores, selected_indices  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

    def plot_boxes(self, scores, xyxy, image, classes,labels):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        Emitir=[NULL]
        imH, imW, _ = image.shape 
        for i in range(len(scores)):
            if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
                xmin = int(max(1,(xyxy[i][0] * imW)))
                ymin = int(max(1,(xyxy[i][1] * imH)))
                xmax = int(min(imH,(xyxy[i][2] * imW)))
                ymax = int(min(imW,(xyxy[i][3] * imH)))
                
                if config.ViewActivate==4:
                    medium_Point_x = int((xmin + xmax)/2)
                    medium_Point_y = int((ymin + ymax)/2)
                    scale_x=medium_Point_x
                    scale_y=medium_Point_y
                    Emitir=[self.class_to_label(labels[i]), scores[i], [scale_x, scale_y]]
                    print(str(Emitir))
                elif config.ViewActivate==3:
                    bgr = (0, 255, 0)
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    Emitir=[self.class_to_label(labels[i]), scores[i], [0, 0]]
                self.ObjectsDetect.emit(Emitir)  
        return image

    def stop(self):
        self.ThreadActive = False
        self.quit()