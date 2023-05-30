'''
xgo图形化python库
'''
import cv2
import cv2 as cv
import copy
import argparse
import numpy as np
import mediapipe as mp
import shutil,requests
import urllib.request
import math
import os,sys,time,logging
import spidev as SPI
import LCD_2inch
import onnxruntime 
import RPi.GPIO as GPIO
from PIL import Image,ImageDraw,ImageFont
from ctypes import c_void_p
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
from pyexpat import model
from keras.models import load_model
import json
from xgolib import XGO


__versinon__ = '1.1.0'
__last_modified__ = '2023/3/30'

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

display = LCD_2inch.LCD_2inch()
display.Init()
display.clear()
splash = Image.new("RGB",(320,240),"black")
display.ShowImage(splash)

#字体载入
font1 = ImageFont.truetype("/home/pi/xgoEdu/Font/msyh.ttc",15)
#情绪识别
face_classifier=cv2.CascadeClassifier('/home/pi/xgoEdu/model/haarcascade_frontalface_default.xml')
classifier = load_model('/home/pi/xgoEdu/model/EmotionDetectionModel.h5')
class_labels=['Angry','Happy','Neutral','Sad','Surprise']

#年纪及性别识别
# 网络模型  和  预训练模型
faceProto = "/home/pi/xgoEdu/model/opencv_face_detector.pbtxt"
faceModel = "/home/pi/xgoEdu/model/opencv_face_detector_uint8.pb"

ageProto = "/home/pi/xgoEdu/model/age_deploy.prototxt"
ageModel = "/home/pi/xgoEdu/model/age_net.caffemodel"

genderProto = "/home/pi/xgoEdu/model/gender_deploy.prototxt"
genderModel = "/home/pi/xgoEdu/model/gender_net.caffemodel"

# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# 加载网络
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
# 人脸检测的网络和模型
faceNet = cv.dnn.readNet(faceModel, faceProto)
padding = 20

cap =cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

'''
人脸检测
'''
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),8)  
    return frameOpencvDnn, bboxes

'''
手势识别函数
'''
def hand_pos(angle):
    pos = None
    # 大拇指角度
    f1 = angle[0]
    # 食指角度
    f2 = angle[1]
    # 中指角度
    f3 = angle[2]
    # 无名指角度
    f4 = angle[3]
    # 小拇指角度
    f5 = angle[4]
    if f1 < 50 and (f2 >= 50 and (f3 >= 50 and (f4 >= 50 and f5 >= 50))):
        pos = 'Good!'
    elif f1 < 50 and (f2 >= 50 and (f3 < 50 and (f4 < 50 and f5 < 50))):
        pos = 'Ok!'
    elif f1 < 50 and (f2 < 50 and (f3 >= 50 and (f4 >= 50 and f5 < 50))):
        pos = 'Rock!'
    elif f1 >= 50 and (f2 >= 50 and (f3 >= 50 and (f4 >= 50 and f5 >= 50))):
        pos = 'stone'
    elif f1 >= 50 and (f2 < 50 and (f3 >= 50 and (f4 >= 50 and f5 >= 50))):
        pos = '1'
    elif f1 >= 50 and (f2 < 50 and (f3 < 50 and (f4 < 50 and f5 >= 50))):
        pos = '3'
    elif f1 >= 50 and (f2 < 50 and (f3 < 50 and (f4 < 50 and f5 < 50))):
        pos = '4'
    elif f1 < 50 and (f2 < 50 and (f3 < 50 and (f4 < 50 and f5 < 50))):
        pos = '5'
    elif f1 >= 50 and (f2 < 50 and (f3 < 50 and (f4 >= 50 and f5 >= 50))):
        pos = '2'
    return pos
def color(value):
  digit = list(map(str, range(10))) + list("ABCDEF")
  value = value.upper()
  if isinstance(value, tuple):
    string = '#'
    for i in value:
      a1 = i // 16
      a2 = i % 16
      string += digit[a1] + digit[a2]
    return string
  elif isinstance(value, str):
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (a3, a2, a1)

'''
竞赛用
'''
class XgoExtend(XGO):
    #def __init__(self, port, screen, button, cap, baud=115200, version='xgomini'):
    def __init__(self, port, baud=115200, version='xgomini'):
        super().__init__(port, baud, version)
        self.reset()
        time.sleep(0.5)
        self.init_yaw = self.read_yaw()
        self.calibration = {}
        with open("/home/pi/xgoEdu/model/calibration.json", 'r', encoding='utf-8') as f:
            self.k = json.load(f)
        self.block_k1 = self.k["BLOCK_k1"]
        self.block_k2 = self.k["BLOCK_k2"]
        self.block_b = self.k["BLOCK_b"]
        self.block_ky = self.k["BLOCK_ky"]
        self.block_by = self.k["BLOCK_by"]
        self.cup_k1 = self.k["CUP_k1"]
        self.cup_k2 = self.k["CUP_k2"]
        self.cup_b = self.k["CUP_b"]

    def show_img(self, img):
        imgok = Image.fromarray(img, mode='RGB')
        #self.screen.ShowImage(imgok)
        display.ShowImage(imgok)
    def check_quit(self):
        button = XGOEDU()
        if button.xgoButton("b"):
            #关闭摄像头并释放对象
            #self.cap.release()
            cap.release()
            #关闭窗口
            cv2.destroyAllWindows()
            #退出系统
            sys.exit()

    def move_by(self, distance, vx, vy, k, mintime):
        runtime = k * abs(distance) + mintime
        self.move_x(math.copysign(vx, distance))
        self.move_y(math.copysign(vy, distance))
        time.sleep(runtime)
        self.move_x(0)
        self.move_y(0)
        time.sleep(0.2)

    def move_x_by(self, distance, vx=18, k=0.035, mintime=0.55):
        self.move_by(distance, vx, 0, k, mintime)
        pass

    def move_y_by(self, distance, vy=18, k=0.0373, mintime=0.5):
        self.move_by(distance, 0, vy, k, mintime)
        pass

    def adjust_x(self, distance, vx=18, k=0.045, mintime=0.6):
        self.move_by(distance, vx, 0, k, mintime)
        pass

    def adjust_y(self, distance, vy=18, k=0.0373, mintime=0.5):
        self.move_by(distance, 0, vy, k, mintime)
        pass

    def adjust_yaw(self, theta, vyaw=16, k=0.08, mintime=0.5):
        runtime = abs(theta) * k + mintime
        self.turn(math.copysign(vyaw, theta))
        time.sleep(runtime)
        self.turn(0)
        pass

    def turn_to(self, theta, vyaw=60, emax=10):
        cur_yaw = self.read_yaw()
        des_yaw = self.init_yaw + theta
        while abs(des_yaw - cur_yaw) >= emax:
            self.turn(math.copysign(vyaw, des_yaw - cur_yaw))
            cur_yaw = self.read_yaw()
            print(cur_yaw)
        self.turn(0)
        time.sleep(0.2)
        pass

    def prepare_for_block(self, x, y, angle, des_x=14, emax_x=1.8, emax_y=1.9, emax_yaw=3.5):
        e_x = x - des_x
        if angle > emax_yaw:
            self.adjust_yaw(-angle)
            # if y < 4 and x > 16.5:
            #     time.sleep(0.3)
            #     self.adjust_y(2)
        elif angle < -emax_yaw:
            self.adjust_yaw(-angle)
            # if y > -4 and x > 16.5:
            #     time.sleep(0.3)
            #     self.adjust_y(-2)
        else:
            if abs(y) > emax_y:
                self.adjust_y(-y)
            else:
                if abs(e_x) > emax_x:
                    self.adjust_x(e_x)
                else:
                    print("DONE BLOCK")
                    self.action(0x83)
                    time.sleep(7)
                    self.reset()
                    time.sleep(0.5)
                    return True
        return False

    def prepare_for_cup(self, x1, x2, y1, y2, vx_k, des_x=16, emax_y=1.8):
        if abs(y1 + y2) > emax_y:
            self.adjust_y(-(y1 + y2) / 2)
            time.sleep(0.3)
        else:
            if 23 < (x1 + x2) / 2 < 60:  # 过滤掉误识别数据
                self.move_x_by((x1 + x2) / 2 - des_x, k=vx_k, mintime=0.65)
                print("DONE CUP")
                self.action(0x84)
                time.sleep(7)
                return True
        return False

    def cal_block(self, s_x, s_y):
        # k1 = 0.00323
        # k2 = -1.272
        # b = 139.5
        # # k1 = 0.002875
        # # k2 = -1.061
        ky = 0.00574
        # b = 108.1
        # x = k1 * s_x * s_x + k2 * s_x + b
        # y = (ky * x + 0.01) * (s_y - 160)
        x = self.block_k1 * s_x * s_x + self.block_k2 * s_x + self.block_b
        # y = self.block_ky * (s_y - 160) * x + self.block_by
        y = (ky * x + 0.01) * (s_y - 160)
        return x, y

    def cal_cup(self, width1, width2, cup_y1, cup_y2):
        kw1 = 1.453e-05
        kw2 = - 1.461e-05
        kc1 = 0.0146
        kc2 = -1.81
        ky = 0.006418
        by = -0.007943
        bc = 77.71
        # 横向畸变
        # kwidth1 = kw1 * (cup_y1 - 160) * (cup_y1 - 160) - kw2 * abs(cup_y1 - 160) + 1
        # kwidth2 = kw1 * (cup_y2 - 160) * (cup_y2 - 160) - kw2 * abs(cup_y2 - 160) + 1
        # width1 = width1 / kwidth1
        # width2 = width2 / kwidth2
        x1 = self.cup_k1 * width1 * width1 + self.cup_k2 * width1 + self.cup_b
        x2 = self.cup_k1 * width2 * width2 + self.cup_k2 * width2 + self.cup_b
        y1 = (ky * x1 - by) * (cup_y1 - 160)
        y2 = (ky * x2 - by) * (cup_y2 - 160)
        return x1, x2, y1, y2

    def get_color_mask(self, color):
        # if color == 'red':
        #     color_lower = np.array([173, 90, 46])
        #     color_upper = np.array([183, 255, 255])
        # elif color == 'green':
        #     color_lower = np.array([73, 150, 70])
        #     color_upper = np.array([88, 255, 255])
        # elif color == 'blue':
        #     color_lower = np.array([100, 100, 50])
        #     color_upper = np.array([110, 255, 255])
        if color == 'red':
            color_lower = (0, 145, 132)
            color_upper = (255, 255, 255)
        elif color == 'green':
            color_lower = (40, 0, 130)
            color_upper = (220, 110, 230)
        elif color == 'blue':
            color_lower = (0, 0, 0)
            color_upper = (255, 136, 120)
        return color_upper, color_lower

    def filter_img(self, frame, color):
        # if color == 'green':
        #     frame_gb = cv2.GaussianBlur(frame, (3, 3), 1)
        #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #     color_upper, color_lower = self.get_color_mask(color)
        #     mask = cv2.inRange(hsv, color_lower, color_upper)
        # else:
        frame = cv2.GaussianBlur(frame, (3, 3), 1)
        color_upper, color_lower = self.get_color_mask(color)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(frame_lab, color_lower, color_upper)

        img_mask = cv2.bitwise_and(frame, frame, mask=mask)
        return img_mask

    def detect_contours(self, frame, color):
        img_mask = self.filter_img(frame, color)
        CANNY_THRESH_1 = 15
        CANNY_THRESH_2 = 120
        edges = cv2.Canny(img_mask, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, img_mask

    def detect_block(self, contours):
        flag = False
        length, width, angle, s_x, s_y = 0, 0, 0, 0, 0
        for i in range(0, len(contours)):
            if cv2.contourArea(contours[i]) < 20 ** 2:
                continue
            rect = cv2.minAreaRect(contours[i])
            if 0.5 < rect[1][0] / rect[1][1] < 2:
                continue
            if not flag:
                if rect[2] > 45:
                    length = rect[1][0]
                    width = rect[1][1]
                    angle = rect[2] - 90
                else:
                    length = rect[1][1]
                    width = rect[1][0]
                    angle = rect[2]
                s_x = rect[0][1]  # s_代表屏幕坐标系
                s_y = rect[0][0]
                flag = True
            else:  # 识别出两个及以上的矩形退出
                flag = False
                break
        return flag, length, width, angle, s_x, s_y

    def detect_cup(self, contours):
        num = 0
        width1, width2, s_y1, s_y2 = 0, 0, 0, 0
        index = [0, 0]
        flag = True
        for i in range(0, len(contours)):
            if cv2.contourArea(contours[i]) < 15 ** 2:
                continue
            rect = cv2.minAreaRect(contours[i])
            if 0.5 < rect[1][0] / rect[1][1] < 2:
                if num == 2:
                    flag = False
                    break
                index[num] = i
                num += 1
        if flag and num == 2:
            c1 = contours[index[0]]
            c2 = contours[index[1]]
            rect1 = cv2.minAreaRect(c1)
            rect2 = cv2.minAreaRect(c2)
            if rect1[2] > 45:
                width1 = rect1[1][1]
            else:
                width1 = rect1[1][0]

            if rect2[2] > 45:
                width2 = rect2[1][1]
            else:
                width2 = rect2[1][0]
            s_y1 = rect1[0][0]
            s_y2 = rect2[0][0]
        else:
            flag = False
        return flag, width1, width2, s_y1, s_y2

    '''
    def detect_single_cup(self,contours):
    '''
    def detect_single_cup(self, contours):
        flag = False
        length, width, angle, s_x, s_y = 0, 0, 0, 0, 0
        for i in range(0, len(contours)):
            if cv2.contourArea(contours[i]) < 15 ** 2:
                continue
            rect = cv2.minAreaRect(contours[i])
            if not (0.5 < rect[1][0] / rect[1][1] < 2):
                continue
            if not flag:
                if rect[2] > 45:
                    length = rect[1][0]
                    width = rect[1][1]
                    angle = rect[2] - 90
                else:
                    length = rect[1][1]
                    width = rect[1][0]
                    angle = rect[2]
                s_x = rect[0][1]
                s_y = rect[0][0]
                flag = True
            else:
                flag = False
                break
        return flag, width, s_y
    
    def search_for_block(self, color, COUNT_MAX=25):
        count = 0
        length, width, angle, s_x, s_y = 0, 0, 0, 0, 0
        x, y = 0, 0
        self.attitude('p', 10)
        while True:
            #ret, frame = self.cap.read()
            ret, frame = cap.read()
            self.check_quit()
            contours, img = self.detect_contours(frame, color)
            flag, temp_length, temp_width, temp_angle, temp_s_x, temp_s_y = self.detect_block(contours)
            if not flag:
                self.show_img(img)
                continue

            cv2.putText(img, '%4.1f' % temp_s_x, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % temp_s_y, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % temp_angle, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % x, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            self.show_img(img)
            count += 1
            length = (count - 1) * length / count + temp_length / count
            width = (count - 1) * width / count + temp_width / count
            angle = (count - 1) * angle / count + temp_angle / count
            s_x = (count - 1) * s_x / count + temp_s_x / count
            s_y = (count - 1) * s_y / count + temp_s_y / count
            if count == COUNT_MAX:
                count = 0
                x, y = self.cal_block(s_x, s_y)
                print("block position x: %4.1f, y: %4.1f" % (x, y))
                done = self.prepare_for_block(x, y, angle)
                if done:
                    break

    '''
    def search_for_block_two_color(self,color1,color2,COUNT_MAX=25):
    '''
    def search_for_block_two_color(self, color1, color2, COUNT_MAX=25):
        count = 0
        length, width, angle, s_x, s_y = 0, 0, 0, 0, 0
        x, y = 0, 0
        self.attitude('p', 10)
        while True:
            ret, frame = cap.read()
            self.check_quit()
            contours, img = self.detect_contours(frame, color1)
            flag, temp_length, temp_width, temp_angle, temp_s_x, temp_s_y = self.detect_block(contours)
            if not flag:
                contours, img = self.detect_contours(frame, color2)
                flag, temp_length, temp_width, temp_angle, temp_s_x, temp_s_y = self.detect_block(contours)
                if not flag:
                    self.show_img(img)
                    continue
    
    def search_for_cup(self, color, COUNT_MAX=25, direction=0, k=0.035):
        count = 0
        width1, width2, s_y1, s_y2 = 0, 0, 0, 0
        x1, x2, y1, y2 = 0, 0, 0, 0
        while True:
            self.check_quit()
            #ret, frame = self.cap.read()
            ret, frame = cap.read()
            contours, img = self.detect_contours(frame, color)
            flag, temp_width1, temp_width2, temp_s_y1, temp_s_y2 = self.detect_cup(contours)
            if not flag:
                self.show_img(img)
                continue

            # cv2.putText(img, '%4.1f' % temp_width1, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            # cv2.putText(img, '%4.1f' % temp_width2, (90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            # cv2.putText(img, '%4.1f' % temp_s_y1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            # cv2.putText(img, '%4.1f' % temp_s_y2, (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % x1, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % x2, (90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y2, (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            self.show_img(img)

            count += 1
            width1 = (count - 1) * width1 / count + temp_width1 / count
            width2 = (count - 1) * width2 / count + temp_width2 / count
            s_y1 = (count - 1) * s_y1 / count + temp_s_y1 / count
            s_y2 = (count - 1) * s_y2 / count + temp_s_y2 / count
            if count == COUNT_MAX:
                count = 0
                x1, x2, y1, y2 = self.cal_cup(width1, width2, s_y1, s_y2)
                print("x1: %4.2f, x2: %4.2f, y1: %4.2f, y2: %4.2f" % (x1, x2, y1, y2))
                done = False
                done = self.prepare_for_cup(x1, x2, y1, y2, vx_k=k)
                if direction != 0:
                    self.turn_to(direction, vyaw=30, emax=2)
                if done:
                    break
    '''
    def search_for_cup_two_color(self,color1,color2,COUNT_MAX=25):
    '''
    def search_for_cup_two_color(self, color1, color2, COUNT_MAX=25, direction=0, k=0.035):
        count = 0
        width1, width2, s_y1, s_y2 = 0, 0, 0, 0
        x1, x2, y1, y2 = 0, 0, 0, 0
        while True:
            self.check_quit()
            ret, frame = cap.read()
            contours, img = self.detect_contours(frame, color1)
            flag, temp_width1, temp_width2, temp_s_y1, temp_s_y2 = self.detect_cup(contours)
            if not flag:
                contours, img = self.detect_contours(frame, color2)
                flag, temp_width1, temp_width2, temp_s_y1, temp_s_y2 = self.detect_cup(contours)
                if not flag:
                    self.show_img(img)
                    continue
            cv2.putText(img, '%4.1f' % x1, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % x2, (90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y2, (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            self.show_img(img)

            count += 1
            width1 = (count - 1) * width1 / count + temp_width1 / count
            width2 = (count - 1) * width2 / count + temp_width2 / count
            s_y1 = (count - 1) * s_y1 / count + temp_s_y1 / count
            s_y2 = (count - 1) * s_y2 / count + temp_s_y2 / count
            if count == COUNT_MAX:
                count = 0
                x1, x2, y1, y2 = self.cal_cup(width1, width2, s_y1, s_y2)
                print("x1: %4.2f, x2: %4.2f, y1: %4.2f, y2: %4.2f" % (x1, x2, y1, y2))
                done = False
                done = self.prepare_for_cup(x1, x2, y1, y2, vx_k=k)
                if direction != 0:
                    self.turn_to(direction, vyaw=30, emax=2)
                if done:
                    break
    '''
    def search_for_cip_CQ(self,color1,color2,COUNT_MAX=25,direction=0,k=0.035)
    '''
    def search_for_cup_CQ(self, color1, color2, COUNT_MAX=25, direction=0, k=0.035):
        count = 0
        width1, width2, s_y1, s_y2 = 0, 0, 0, 0
        x1, x2, y1, y2 = 0, 0, 0, 0
        while True:
            self.check_quit()
            ret, frame = cap.read()
            contours, img = self.detect_contours(frame, color1)
            flag, temp_width1, temp_s_y1 = self.detect_single_cup(contours)
            if not flag:
                self.show_img(img)
                continue

            ret, frame = cap.read()
            contours, img = self.detect_contours(frame, color2)
            flag, temp_width2, temp_s_y2 = self.detect_single_cup(contours)
            if not flag:
                self.show_img(img)
                continue
            cv2.putText(img, '%4.1f' % x1, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % x2, (90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            cv2.putText(img, '%4.1f' % y2, (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            self.show_img(img)

            count += 1
            width1 = (count - 1) * width1 / count + temp_width1 / count
            width2 = (count - 1) * width2 / count + temp_width2 / count
            s_y1 = (count - 1) * s_y1 / count + temp_s_y1 / count
            s_y2 = (count - 1) * s_y2 / count + temp_s_y2 / count
            if count == COUNT_MAX:
                count = 0
                x1, x2, y1, y2 = self.cal_cup(width1, width2, s_y1, s_y2)
                print("x1: %4.2f, x2: %4.2f, y1: %4.2f, y2: %4.2f" % (x1, x2, y1, y2))
                done = False
                done = self.prepare_for_cup(x1, x2, y1, y2, vx_k=k)
                if direction != 0:
                    self.turn_to(direction, vyaw=30, emax=2)
                if done:
                    break
    def calibration_block(self, color, COUNT_MAX=25):
        count = 0
        block_num = 0
        state = 0
        path = 'calibration.json'
        s_x = 0
        s_y = 0
        s_x_list = [186, 174.5, 163.5, 150.5, 143.5, 138.5, 131.5, 125.6]
        x_list = [13, 15, 17, 20, 22, 24, 27, 30]

        ky_list = []
        y_list = [2.25, 0.25, -1.75, 0.25, 2.25, 0.25, -1.75, 0.25]
        kx_list = []
        #机器狗俯身
        self.attitude('p', 10)
        while True:
            self.check_quit()
            #ret为是否补货成功，frame为捕获的每一帧图像
            #ret, frame = self.cap.read()
            ret, frame = cap.read()
            contours, img = self.detect_contours(frame, color)
            if state == 0:
                cv2.putText(img, 'Put BLOCK in', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 200), 2)
                cv2.putText(img, '- ' + str(block_num + 1) + ' -', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (200, 0, 200), 2)
                cv2.putText(img, 'Then press D(up right)', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                self.show_img(img)
                button = XGOEDU()
                if button.xgoButton("d"):
                    state = 1
                    time.sleep(0.5)
            elif state == 1:
                flag, temp_length, temp_width, temp_angle, temp_s_x, temp_s_y = self.detect_block(contours)
                if not flag:
                    self.show_img(img)
                    continue
                cv2.putText(img, 'Detecting......', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)
                cv2.putText(img, '%4.1f' % temp_s_x, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                cv2.putText(img, '%4.1f' % temp_s_y, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                self.show_img(img)
                count += 1
                s_x = s_x * (count - 1) / count + temp_s_x / count
                s_y = s_y * (count - 1) / count + temp_s_y / count
                if count == COUNT_MAX:
                    count = 0
                    s_x_list.append(s_x)
                    s_x_list.append(s_x)
                    ky_list.append(y_list[block_num])
                    kx_list.append((s_y - 160) * (14 + block_num * 3))
                    x_list.append(14 + block_num * 3)
                    x_list.append(14 + block_num * 3)
                    block_num += 1
                    state = 0
                    print("Finish" + str(block_num))
                    if block_num == 8:
                        z = np.polyfit(s_x_list, x_list, 2)
                        self.calibration["BLOCK_k1"] = z[0]
                        self.calibration["BLOCK_k2"] = z[1]
                        self.calibration["BLOCK_b"] = z[2]
                        z = np.polyfit(kx_list, ky_list, 1)
                        self.calibration["BLOCK_ky"] = z[0]
                        self.calibration["BLOCK_by"] = z[1]
                        break

    def calibration_cup(self, color, COUNT_MAX=25):
        count = 0
        cap_num = 0
        state = 0
        path = 'calibration.json'
        width1 = 0
        width2 = 0
        width_list = []
        x_list = []

        while True:
            self.check_quit()
            #ret, frame = self.cap.read()
            ret, frame = cap.read()
            contours, img = self.detect_contours(frame, color)
            if state == 0:
                cv2.putText(img, 'Put Two Cups In', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 200), 2)
                cv2.putText(img, '- ' + str(cap_num + 1) + ' -', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (200, 0, 200), 2)
                cv2.putText(img, 'Then press D(up right)', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                self.show_img(img)
                button = XGOEDU()
                if button.xgoButton("d"):
                    state = 1
                    time.sleep(0.5)
            elif state == 1:
                flag, temp_width1, temp_width2, temp_s_y1, temp_s_y2 = self.detect_cup(contours)
                if not flag:
                    self.show_img(img)
                    continue
                cv2.putText(img, 'Detecting......', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)
                self.show_img(img)
                count += 1
                width1 = width1 * (count - 1) / count + temp_width1 / count
                width2 = width2 * (count - 1) / count + temp_width2 / count
                if count == COUNT_MAX:
                    count = 0
                    width_list.append(width1)
                    width_list.append(width2)
                    x_list.append(24.7 + cap_num * 2)
                    x_list.append(24.7 + cap_num * 2)
                    cap_num += 1
                    state = 0
                    print("Finish" + str(cap_num))
                    if cap_num == 8:
                        z = np.polyfit(width_list, x_list, 2)
                        self.calibration["CUP_k1"] = z[0]
                        self.calibration["CUP_k2"] = z[1]
                        self.calibration["CUP_b"] = z[2]
                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(self.calibration, f)
                        break

    def calibration_contest(self, color='red'):
        self.calibration_block(color)
        self.reset()
        time.sleep(1)
        self.calibration_cup(color)

    def show_filter_img(self, color):
        while True:
            #ret, frame = self.cap.read()
            ret, frame = cap.read()
            img = self.filter_img(frame, color)
            self.show_img(img)
            self.check_quit()

class XGOEDU():
    def __init__(self):
        self.key1=17
        self.key2=22
        self.key3=23
        self.key4=24
        GPIO.setup(self.key1,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(self.key2,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(self.key3,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(self.key4,GPIO.IN,GPIO.PUD_UP)
    #画布初始化
    def lcd_init(self,color):
        if color == "black":
            splash = Image.new("RGB",(320,240),"black")
        elif color == "white":
            splash = Image.new("RGB",(320,240),"white")
        elif color == "red":
            splash = Image.new("RGB",(320,240),"red") 
        elif color == "green":
            splash = Image.new("RGB",(320,240),"green")
        elif color == "blue":
            splash = Image.new("RGB",(320,240),"blue")
        display.ShowImage(splash)
    #绘画直线
    '''
    x1,y1为初始点坐标,x2,y2为终止点坐标
    '''
    def lcd_line(self,x1,y1,x2,y2):
        draw = ImageDraw.Draw(splash)
        draw.line([(x1,y1),(x2,y2)],fill = "WHITE",width = 2)
        display.ShowImage(splash)
    #绘画圆
    '''
    x1,y1,x2,y2为定义给定边框的两个点,angle0为初始角度,angle1为终止角度
    '''
    def lcd_circle(self,x1,y1,x2,y2,angle0,angle1):
        draw = ImageDraw.Draw(splash)
        draw.arc((x1,y1,x2,y2),angle0,angle1,fill=(255,255,255),width = 2)
        display.ShowImage(splash)
    #绘画矩形
    '''
    x1,y1为初始点坐标,x2,y2为对角线终止点坐标
    '''
    def lcd_rectangle(self,x1,y1,x2,y2):
        draw = ImageDraw.Draw(splash)
        draw.rectangle((x1,y1,x2,y2),fill = None,outline = "WHITE",width = 2)
        display.ShowImage(splash)
    #清除屏幕
    def lcd_clear(self):
        splash = Image.new("RGB",(320,240),"black")
        display.ShowImage(splash)
    #显示图片
    '''
    图片的大小为320*240,jpg格式
    '''
    def lcd_picture(self,filename):
        image = Image.open(filename)
        display.ShowImage(image)
    #显示文字
    '''
    font1为载入字体,微软雅黑
    目前支持英文和数字，暂不支持中文
    '''
    def lcd_text(self,x1,y1,content):
        draw = ImageDraw.Draw(splash)
        draw.text((x1,y1),content,fill = "WHITE",font=font1)
        display.ShowImage(splash)
    #key_value
    '''
    a左上按键
    b右上按键
    c左下按键
    d右下按键
    返回值 0未按下,1按下
    '''
    def xgoButton(self,Button):
        if Button == "a":
            last_state_a =GPIO.input(self.key1)
            time.sleep(0.02)
            return(not last_state_a)
        elif Button == "b":
            last_state_b=GPIO.input(self.key2)
            time.sleep(0.02)
            return(not last_state_b)
        elif Button == "c":
            last_state_c=GPIO.input(self.key3)
            time.sleep(0.02)
            return(not last_state_c)
        elif Button == "d":
            last_state_d=GPIO.input(self.key4)
            time.sleep(0.02)
            return(not last_state_d)
    #speaker
    '''
    filename 文件名 字符串
    '''
    def xgoSpeaker(self,filename):
        os.system("mplayer"+" "+filename)
    #audio_record
    '''
    filename 文件名 字符串
    seconds 录制时间S 字符串
    '''
    def xgoAudioRecord(self,filename,seconds):
        command1 = "sudo arecord -D hw:1,0 -d"
        command2 = "-f S32_LE -r 16000 -c 2"
        os.system(command1+" "+seconds+" "+command2+" "+filename)
    '''
    开启摄像头
    '''
    def cameraOn(self):
        while True:
            success,image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue
            #cv2.imshow('frame',image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            imgok = Image.fromarray(image)
            display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    '''
    开启摄像头并拍照
    '''
    def takePhoto(self):
        while True:
            success,image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue
            cv2.imshow('frame',image)
            cv2.imwrite('/home/pi/xgoEdu/camera/file.jpg',image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            imgok = Image.fromarray(image)
            display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    '''
    手势识别
    '''
    def gestureRecognition(self):
        hand = hands(0,2,0.6,0.5)
        while True:
            success,image = cap.read()
            datas = hand.run(image)
            cv2.imshow('OpenCV',image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            if not success:
                print("Ignoring empty camera frame")
                continue
            for data in datas:
                pos_left = ''
                pos_right = ''
                rect = data['rect']
                right_left = data['right_left']
                center = data['center']
                dlandmark = data['dlandmark']
                hand_angle = data['hand_angle']
                XGOEDU.rectangle(self,image,rect,"#33cc00",2)
                #XGOEDU.text(self,image,right_left,center,2,"#cc0000",5)
                if right_left == 'L':
                    XGOEDU.text(self,image,hand_pos(hand_angle),(180,80),1.5,"#33cc00",2)
                    pos_left = hand_pos(hand_angle)
                elif right_left == 'R':
                    XGOEDU.text(self,image,hand_pos(hand_angle),(50,80),1.5,"#ff0000",2)
                    pos_right = hand_pos(hand_angle)
                for i in dlandmark:
                    XGOEDU.circle(self,image,i,3,"#ff9900",-1)
            imgok = Image.fromarray(image)
            display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            #return(pos_left,pos_right)
    '''
    yolo
    '''
    def yoloFast(self):
        yolo = yoloXgo('/home/pi/xgoEdu/model/Model.onnx',
        ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'],
        [352,352],0.6)
        while True:
            success,image = cap.read()
            datas = yolo.run(image)
            cv2.imshow('OpenCV',image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            if not success:
                print("Ignoring empty camera frame")
                continue
            if datas:
                for data in datas:
                    XGOEDU.rectangle(self,image,data['xywh'],"#33cc00",2)
                    xy= (data['xywh'][0], data['xywh'][1])
                    XGOEDU.text(self,image,data['classes'],xy,1,"#ff0000",2)
                    value_yolo = data['classes']
            imgok = Image.fromarray(image)
            display.ShowImage(imgok)
            #return(value_yolo)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    '''
    人脸坐标点检测
    '''
    def face_detect(self):
        face = face_detection(0.7)
        while True:
            success,image = cap.read()
            datas = face.run(image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            if not success:
                print("Ignoring empty camera frame")
                continue
            for data in datas:
                print(data)
                lefteye = str(data['left_eye'])
                righteye = str(data['right_eye'])
                nose = str(data['nose'])
                mouth = str(data['mouth'])
                leftear = str(data['left_ear'])
                rightear = str(data['right_ear'])
                cv2.putText(image,'lefteye',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                cv2.putText(image,lefteye,(100,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                cv2.putText(image,'righteye',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.putText(image,righteye,(100,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.putText(image,'nose',(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(image,nose,(100,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(image,'leftear',(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                cv2.putText(image,leftear,(100,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                cv2.putText(image,'rightear',(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,0,200),2)
                cv2.putText(image,rightear,(100,110),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,0,200),2)
                XGOEDU.rectangle(self,image,data['rect'],"#33cc00",2)
            #cv2.imshow('OpenCV',image)
            imgok = Image.fromarray(image)
            display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    '''
    情绪识别
    '''
    def emotion(self):
        while True:
            success,image=cap.read()
            labels=[]
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)
            label=''
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi=roi_gray.astype('float')/255.0
                    roi=img_to_array(roi)
                    roi=np.expand_dims(roi,axis=0)

                    preds=classifier.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    print(label)
                    label_position=(x,y)
                else:
                    pass
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image, 1)
            try:
                cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            except:
                pass
            imgok = Image.fromarray(image)
            display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    '''
    年纪及性别检测
    '''
    def agesex(self):
        while True:
            t = time.time()
            hasFrame,image = cap.read()
            image = cv.flip(image, 1)
            frameFace, bboxes = getFaceBox(faceNet, image)
            if not bboxes:
                print("No face Detected, Checking next frame")
            gender=''
            age=''
            for bbox in bboxes:
                face = image[max(0, bbox[1] - padding):min(bbox[3] + padding, image.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, image.shape[1] - 1)]
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)   
                genderPreds = genderNet.forward()   
                gender = genderList[genderPreds[0].argmax()]  
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                label = "{},{}".format(gender, age)
                cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,cv.LINE_AA)  
            b,g,r = cv2.split(frameFace)
            frameFace = cv2.merge((r,g,b))
            imgok = Image.fromarray(frameFace)
            display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    
    def rectangle(self,frame,z,colors,size):
        frame=cv2.rectangle(frame,(int(z[0]),int(z[1])),(int(z[0]+z[2]),int(z[1]+z[3])),color(colors),size)
        return frame
        
    def circle(self,frame,xy,rad,colors,tk):
        frame=cv2.circle(frame,xy,rad,color(colors),tk)
        return frame
    
    def text(self,frame,text,xy,font_size,colors,size):
        frame=cv2.putText(frame,text,xy,cv2.FONT_HERSHEY_SIMPLEX,font_size,color(colors),size)
        return frame       

class hands():
    def __init__(self,model_complexity,max_num_hands,min_detection_confidence,min_tracking_confidence):
        self.model_complexity = model_complexity
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
    
    def run(self,cv_img):
        image = cv_img
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(image)
        hf=[]
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 手的重心计算
                cx, cy = self.calc_palm_moment(debug_image, hand_landmarks)
                # 手的外接矩形计算
                rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # 手的个关键点
                dlandmark = self.dlandmarks(debug_image,hand_landmarks,handedness)

                hf.append({'center':(cx,cy),'rect':rect,'dlandmark':dlandmark[0],'hand_angle':self.hand_angle(dlandmark[0]),'right_left':dlandmark[1]})
        return hf

    def calc_palm_moment(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        palm_array = np.empty((0, 2), int)
        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            if index == 0:  # 手首1
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:  # 手首2
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:  # 人差指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:  # 中指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:  # 薬指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:  # 小指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
        M = cv.moments(palm_array)
        cx, cy = 0, 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        return cx, cy

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, w, h]

    def dlandmarks(self,image, landmarks, handedness):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append((landmark_x, landmark_y))
        return landmark_point,handedness.classification[0].label[0]

    def vector_2d_angle(self, v1, v2):
        v1_x = v1[0]
        v1_y = v1[1]
        v2_x = v2[0]
        v2_y = v2[1]
        try:
            angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
        except:
            angle_ = 180
        return angle_

    def hand_angle(self,hand_):
        angle_list = []
        # thumb 大拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
            ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
            )
        angle_list.append(angle_)
        # index 食指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
            ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
            )
        angle_list.append(angle_)
        # middle 中指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
            ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
            )
        angle_list.append(angle_)
        # ring 無名指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
            ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
            )
        angle_list.append(angle_)
        # pink 小拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
            ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
            )
        angle_list.append(angle_)
        return angle_list
    
class yoloXgo():
    def __init__(self,model,classes,inputwh,thresh):
        self.session = onnxruntime.InferenceSession(model)
        self.input_width=inputwh[0]
        self.input_height=inputwh[1]
        self.thresh=thresh
        self.classes=classes
        
    def sigmoid(self,x):
        return 1. / (1 + np.exp(-x))

    # tanh函数
    def tanh(self,x):
        return 2. / (1 + np.exp(-2 * x)) - 1

    # 数据预处理
    def preprocess(self,src_img, size):
        output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
        output = output.transpose(2,0,1)
        output = output.reshape((1, 3, size[1], size[0])) / 255
        return output.astype('float32') 

    # nms算法
    def nms(self,dets,thresh=0.45):
        # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
        # #thresh:0.3,0.5....
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
        order = scores.argsort()[::-1]  # 对分数进行倒排序
        keep = []  # 用来保存最后留下来的bboxx下标

        while order.size > 0:
            i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
            keep.append(i)

            # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留ovr小于thresh的bbox，进入下一次迭代。
            inds = np.where(ovr <= thresh)[0]

            # 因为ovr中的索引不包括order[0]所以要向后移动一位
            order = order[inds + 1]
        
        output = []
        for i in keep:
            output.append(dets[i].tolist())

        return output

    def run(self, img,):
        pred = []

        # 输入图像的原始宽高
        H, W, _ = img.shape

        # 数据预处理: resize, 1/255
        data = self.preprocess(img, [self.input_width, self.input_height])

        # 模型推理
        input_name = self.session.get_inputs()[0].name
        feature_map = self.session.run([], {input_name: data})[0][0]

        # 输出特征图转置: CHW, HWC
        feature_map = feature_map.transpose(1, 2, 0)
        # 输出特征图的宽高
        feature_map_height = feature_map.shape[0]
        feature_map_width = feature_map.shape[1]

        # 特征图后处理
        for h in range(feature_map_height):
            for w in range(feature_map_width):
                data = feature_map[h][w]

                # 解析检测框置信度
                obj_score, cls_score = data[0], data[5:].max()
                score = (obj_score ** 0.6) * (cls_score ** 0.4)

                # 阈值筛选
                if score > self.thresh:
                    # 检测框类别
                    cls_index = np.argmax(data[5:])
                    # 检测框中心点偏移
                    x_offset, y_offset = self.tanh(data[1]), self.tanh(data[2])
                    # 检测框归一化后的宽高
                    box_width, box_height = self.sigmoid(data[3]), self.sigmoid(data[4])
                    # 检测框归一化后中心点
                    box_cx = (w + x_offset) / feature_map_width
                    box_cy = (h + y_offset) / feature_map_height
                    
                    # cx,cy,w,h => x1, y1, x2, y2
                    x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                    x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                    x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                    pred.append([x1, y1, x2, y2, score, cls_index])
        datas=np.array(pred)
        data=[]
        if len(datas)>0:
            boxes=self.nms(datas)
            for b in boxes:
                obj_score, cls_index = b[4], int(b[5])
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                s={'classes':self.classes[cls_index],'score':'%.2f' % obj_score,'xywh':[x1,y1,x2-x1,y2-y1],}
                data.append(s)
            return data
        else:
            return False

class face_detection():
    def __init__(self,min_detection_confidence):
        self.model_selection = 0
        self.min_detection_confidence =min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
        )

    def run(self,cv_img):
        image = cv_img
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.face_detection.process(cv_img)
        face=[]
        if results.detections is not None:
            for detection in results.detections:
                data =self.draw_detection(image,detection) 
                face.append(data)
        return face
    def draw_detection(self, image, detection):
        image_width, image_height = image.shape[1], image.shape[0]
        bbox = detection.location_data.relative_bounding_box
        bbox.xmin = int(bbox.xmin * image_width)
        bbox.ymin = int(bbox.ymin * image_height)
        bbox.width = int(bbox.width * image_width)
        bbox.height = int(bbox.height * image_height)


        # 位置：右目
        keypoint0 = detection.location_data.relative_keypoints[0]
        keypoint0.x = int(keypoint0.x * image_width)
        keypoint0.y = int(keypoint0.y * image_height)


        # 位置：左目
        keypoint1 = detection.location_data.relative_keypoints[1]
        keypoint1.x = int(keypoint1.x * image_width)
        keypoint1.y = int(keypoint1.y * image_height)


        # 位置：鼻
        keypoint2 = detection.location_data.relative_keypoints[2]
        keypoint2.x = int(keypoint2.x * image_width)
        keypoint2.y = int(keypoint2.y * image_height)


        # 位置：口
        keypoint3 = detection.location_data.relative_keypoints[3]
        keypoint3.x = int(keypoint3.x * image_width)
        keypoint3.y = int(keypoint3.y * image_height)

        # 位置：右耳
        keypoint4 = detection.location_data.relative_keypoints[4]
        keypoint4.x = int(keypoint4.x * image_width)
        keypoint4.y = int(keypoint4.y * image_height)

        # 位置：左耳
        keypoint5 = detection.location_data.relative_keypoints[5]
        keypoint5.x = int(keypoint5.x * image_width)
        keypoint5.y = int(keypoint5.y * image_height)

        data={'id':detection.label_id[0],
            'score':round(detection.score[0], 3),
            'rect':[int(bbox.xmin),int(bbox.ymin),int(bbox.width),int(bbox.height)],
            'right_eye':(int(keypoint0.x),int(keypoint0.y)),
            'left_eye':(int(keypoint1.x),int(keypoint1.y)),
            'nose':(int(keypoint2.x),int(keypoint2.y)),
            'mouth':(int(keypoint3.x),int(keypoint3.y)),
            'right_ear':(int(keypoint4.x),int(keypoint4.y)),
            'left_ear':(int(keypoint5.x),int(keypoint5.y)),
            }
        return data
