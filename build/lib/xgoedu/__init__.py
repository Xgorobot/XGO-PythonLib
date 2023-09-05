'''
xgo图形化python库  edu库
'''
import cv2
import numpy as np
import math
import os,sys,time,json,base64
import spidev as SPI
import xgoscreen.LCD_2inch as LCD_2inch
import RPi.GPIO as GPIO
from PIL import Image,ImageDraw,ImageFont
import json
import threading
# from xgolib import XGO
# from keras.preprocessing import image
# import _thread  使用_thread会报错，坑！


__versinon__ = '1.3.6'
__last_modified__ = '2023/9/5'

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

camera_still=False


'''
人脸检测
'''
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
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
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),8)  
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
        pos = 'Good'
    elif f1 < 50 and (f2 >= 50 and (f3 < 50 and (f4 < 50 and f5 < 50))):
        pos = 'Ok'
    elif f1 < 50 and (f2 < 50 and (f3 >= 50 and (f4 >= 50 and f5 < 50))):
        pos = 'Rock'
    elif f1 >= 50 and (f2 >= 50 and (f3 >= 50 and (f4 >= 50 and f5 >= 50))):
        pos = 'Stone'
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



class XGOEDU():
    def __init__(self):
        self.display = LCD_2inch.LCD_2inch()
        self.display.Init()
        self.display.clear()
        self.splash = Image.new("RGB",(320,240),"black")
        self.display.ShowImage(self.splash)
        self.draw = ImageDraw.Draw(self.splash)
        self.font = ImageFont.truetype("/home/pi/model/msyh.ttc",15)
        self.key1=17
        self.key2=22
        self.key3=23
        self.key4=24
        self.cap=None
        self.hand=None
        self.yolo=None
        self.face=None
        self.face_classifier=None
        self.classifier=None
        self.agesexmark=None
        self.camera_still=False
        GPIO.setup(self.key1,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(self.key2,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(self.key3,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(self.key4,GPIO.IN,GPIO.PUD_UP)

    def open_camera(self):
        if self.cap==None:
            self.cap =cv2.VideoCapture(0)
            self.cap.set(3,320)
            self.cap.set(4,240)

    def fetch_token(self):
        from urllib.request import urlopen
        from urllib.request import Request
        from urllib.error import URLError
        from urllib.parse import urlencode
        API_KEY = 'Q4ZgU8bfnhA8HQFnNucBO2ut'
        SECRET_KEY = 'MqFrVgdwoM8ZuGIp0NIFF7qfYti4mjP6'
        TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
        params = {'grant_type': 'client_credentials',
                'client_id': API_KEY,
                'client_secret': SECRET_KEY}
        post_data = urlencode(params)
        post_data = post_data.encode( 'utf-8')
        req = Request(TOKEN_URL, post_data)
        try:
            f = urlopen(req)
            result_str = f.read()
        except URLError as err:
            print('token http response http code : ' + str(err.code))
            result_str = err.read()
        result_str =  result_str.decode()

        #print(result_str)
        result = json.loads(result_str)
        #print(result)
        SCOPE=False
        if ('access_token' in result.keys() and 'scope' in result.keys()):
            #print(SCOPE)
            if SCOPE and (not SCOPE in result['scope'].split(' ')):  # SCOPE = False 忽略检查
                raise DemoError('scope is not correct')
            #print('SUCCESS WITH TOKEN: %s  EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            return result['access_token']
        else:
            raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')



    #绘画直线
    '''
    x1,y1为初始点坐标,x2,y2为终止点坐标
    '''
    def lcd_line(self,x1,y1,x2,y2,color="WHITE",width=2):
        self.draw.line([(x1,y1),(x2,y2)],fill=color ,width=width)
        self.display.ShowImage(self.splash)
    #绘画圆
    '''
    x1,y1,x2,y2为定义给定边框的两个点,angle0为初始角度,angle1为终止角度
    '''
    def lcd_circle(self,x1,y1,x2,y2,angle0,angle1,color="WHITE",width=2):
        self.draw.arc((x1,y1,x2,y2),angle0,angle1,fill=color ,width=width)
        self.display.ShowImage(self.splash)

    #绘画圆:  根据圆形点和半径画圆
    '''
    center_x, center_y 圆心点坐标
    radius 圆半径长度 mm
    
    '''
    def lcd_round(self,center_x, center_y, radius, color, width=2):
        # Calculate the bounding box for the circle
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius
    
        # Call lcd_circle() function to draw the circle
        self.lcd_circle(x1, y1, x2, y2, 0, 360, color=color, width=width)
  

    
    #绘画矩形
    '''
    x1,y1为初始点坐标,x2,y2为对角线终止点坐标
    '''
    def lcd_rectangle(self,x1,y1,x2,y2,fill=None,outline="WHITE",width=2):
        self.draw.rectangle((x1,y1,x2,y2),fill=fill,outline=outline,width=width)
        self.display.ShowImage(self.splash)
    #清除屏幕
    def lcd_clear(self):
        self.splash = Image.new("RGB",(320,240),"black")
        self.draw = ImageDraw.Draw(self.splash)
        self.display.ShowImage(self.splash)
    #显示图片
    '''
    图片的大小为320*240,jpg格式
    '''
    def lcd_picture(self,filename,x=0,y=0):
        path="/home/pi/xgoPictures/"
        image = Image.open(path+filename)
        self.splash.paste(image,(x,y))
        self.display.ShowImage(self.splash)
    #显示文字
    '''
    x1,y1为初始点坐标,content为内容
    '''
    def lcd_text(self,x,y,content,color="WHITE",fontsize=15):
        if fontsize!=15:
            self.font = ImageFont.truetype("/home/pi/model/msyh.ttc",fontsize)
        self.draw.text((x,y),content,fill=color,font=self.font)
        self.display.ShowImage(self.splash)
    #流式显示所有文字
    '''
    x1,y1为初始点坐标,content为内容
    遇到回车符自动换行，遇到边缘换行，一页满了自动清屏，2,2开始继续显示
    '''
    def display_text_on_screen(self, content, color, start_x=2, start_y=2, font_size=20, screen_width=320, screen_height=240):
        # 计算每行可显示字符的数量和行数
        char_width = font_size +1  #// 2
        chars_per_line = screen_width // char_width
        lines = screen_height // char_width
    
        # 拆分内容为逐个字符的列表
        chars = list(content)
     
        # 处理换行符
        line_break_indices = [i for i, char in enumerate(chars) if char == '\n']
    
    
        # 计算总行数和页数
        total_lines = len(chars) // chars_per_line + 1
        total_pages = (total_lines - 1+len(line_break_indices)) // lines + 1
    
        # 清屏
        self.display.clear()
    
        # 逐行显示文字
        current_page = 1
        current_line = 1
        current_char = 0
    
        while current_page <= total_pages or  current_char < len(chars) :
            self.display.clear()
            # 计算当前页要显示的行数
            if current_page < total_pages or  current_char < len(chars) :
                lines_to_display = lines
            else:
                lines_to_display = (total_lines - 1) % lines + 1
    
            current_line = 1
            # 显示当前页的内容
            for line in range(lines_to_display):
                current_x = start_x
                current_y = start_y + current_line * char_width # font_size
                current_line +=1
                if current_line >= lines:
                    break
    
                # 显示当前行的文字
                for _ in range(chars_per_line):
                    # 检查是否所有字符都已显示完毕
                    if current_char >= len(chars):
                        break
    
                    char = chars[current_char]
                    if char == '\n':
                        current_x = start_x
                        current_y = start_y + current_line * char_width # font_size
                        current_line +=1
                       
                        self.lcd_text(current_x, current_y, char, color, font_size)
                        current_char += 1
                        break  # continue
    
                    self.lcd_text(current_x, current_y, char, color, font_size)
                    current_x += char_width
                    current_char += 1
    
                # 检查是否所有字符都已显示完毕
                if current_char >= len(chars):
                    break
    
            # 更新当前页和当前行
            current_page += 1
            current_line += lines_to_display
    
            # 等待显示时间或手动触发翻页
            # 这里可以根据需要添加适当的延时代码或触发翻页的机制
    
        # 如果内容超过一屏幕，则清屏
        # if total_lines > lines:
        if current_page < total_pages:
            self.display.clear()
    
    #key_value
    '''
    a左上按键
    b右上按键
    c左下按键
    d右下按键
    返回值 0未按下,1按下
    '''
    def xgoButton(self,button):
        if button == "a":
            last_state_a =GPIO.input(self.key1)
            time.sleep(0.02)
            return(not last_state_a)
        elif button == "b":
            last_state_b=GPIO.input(self.key2)
            time.sleep(0.02)
            return(not last_state_b)
        elif button == "c":
            last_state_c=GPIO.input(self.key3)
            time.sleep(0.02)
            return(not last_state_c)
        elif button == "d":
            last_state_d=GPIO.input(self.key4)
            time.sleep(0.02)
            return(not last_state_d)
    #speaker
    '''
    filename 文件名 字符串
    '''
    def xgoSpeaker(self,filename):
        path="/home/pi/xgoMusic/"
        os.system("mplayer"+" "+path+filename)

    def xgoVideoAudio(self,filename):
        path="/home/pi/xgoVideos/"
        time.sleep(0.2)  #音画速度同步了 但是时间轴可能不同步 这里调试一下
        cmd="sudo mplayer "+path+filename+" -novideo"
        os.system(cmd)

    def xgoVideo(self,filename):
        path="/home/pi/xgoVideos/"
        x=threading.Thread(target=self.xgoVideoAudio,args=(filename,))
        x.start()
        global counter
        video=cv2.VideoCapture(path+filename)
        print(path+filename)
        fps = video.get(cv2.CAP_PROP_FPS) 
        print(fps)
        init_time=time.time()
        counter=0
        while True:
            grabbed, dst = video.read()
            try:
                b,g,r = cv2.split(dst)
                dst = cv2.merge((r,g,b))
            except:
                pass
            try:
                imgok = Image.fromarray(dst)
            except:
                break
            self.display.ShowImage(imgok)
            #强制卡帧数 实测帧数不要超过20贞 否则显示跟不上 但是20贞转换经常有问题 所以建议直接15贞
            counter += 1
            ctime=time.time()- init_time
            if ctime != 0:
                qtime=counter/fps-ctime
                #print(qtime)
                if qtime>0:
                    time.sleep(qtime)
            if not grabbed:
                break
        
    #audio_record
    '''
    filename 文件名 字符串
    seconds 录制时间S 字符串
    '''
    def xgoAudioRecord(self,filename="record",seconds=5):
        path="/home/pi/xgoMusic/"
        command1 = "sudo arecord -d"
        command2 = "-f S32_LE -r 8000 -c 1 -t wav"
        cmd=command1+" "+str(seconds)+" "+command2+" "+path+filename+".wav"
        print(cmd)
        os.system(cmd)

    def xgoCamera(self,switch):
        global camera_still
        if switch:
            self.open_camera()
            self.camera_still=True
            t = threading.Thread(target=self.camera_mode)  
            t.start() 
        else:
            self.camera_still=False
            time.sleep(0.5)
            splash = Image.new("RGB",(320,240),"black")
            self.display.ShowImage(splash)

    def camera_mode(self):
        self.camera_still=True
        while 1:
            success,image = self.cap.read()
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            imgok = Image.fromarray(image)
            self.display.ShowImage(imgok)
            if not self.camera_still:
                break

    def xgoVideoRecord(self,filename="record",seconds=5):
        path="/home/pi/xgoVideos/"
        self.camera_still=False
        time.sleep(0.6)
        self.open_camera()
        FPS=15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoWrite = cv2.VideoWriter(path+filename+'.mp4', fourcc, FPS, (width,height))
        starttime=time.time()
        while 1:
            print('recording...')
            ret, image = self.cap.read()
            if not ret:
                break
            videoWrite.write(image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            imgok = Image.fromarray(image)
            self.display.ShowImage(imgok)
            if time.time()-starttime>seconds:
                break
        print('recording done')
        self.xgoCamera(True)
        videoWrite.release()

    def xgoTakePhoto(self,filename="photo"):
        path="/home/pi/xgoPictures/"
        self.camera_still=False
        time.sleep(0.6)
        self.open_camera()
        success,image = self.cap.read()
        cv2.imwrite(path+filename+'.jpg',image)
        if not success:
            print("Ignoring empty camera frame")
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        image = cv2.flip(image,1)
        imgok = Image.fromarray(image)
        self.display.ShowImage(imgok)
        print('photo writed!')
        time.sleep(0.7)
        self.xgoCamera(True)


    '''
    开启摄像头  A键拍照 B键录像 C键退出
    '''
    def camera(self,filename="camera"):
        font = ImageFont.truetype("/home/pi/model/msyh.ttc",20)
        self.open_camera()
        while True:
            success,image = self.cap.read()
            #cv2.imwrite('/home/pi/xgoEdu/camera/file.jpg',image)
            if not success:
                print("Ignoring empty camera frame")
                continue
            #cv2.imshow('frame',image)
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
            image = cv2.flip(image,1)
            imgok = Image.fromarray(image)
            self.display.ShowImage(imgok)
            if cv2.waitKey(5) & 0xFF == 27:
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
            if XGOEDU.xgoButton(self,"a"):
                draw=ImageDraw.Draw(imgok)
                cv2.imwrite(filename+'.jpg',image)
                print('photo writed!')
                draw.text((5,5),filename+'.jpg saved!',fill=(255,0,0),font=font)
                self.display.ShowImage(imgok)
                time.sleep(1)
            if XGOEDU.xgoButton(self,"b"):
                FPS=15
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                videoWrite = cv2.VideoWriter(filename+'.mp4', fourcc, FPS, (width,height))
                while 1:
                    ret, image = self.cap.read()
                    if not ret:
                        break
                    videoWrite.write(image)
                    b,g,r = cv2.split(image)
                    image = cv2.merge((r,g,b))
                    image = cv2.flip(image,1)
                    imgok = Image.fromarray(image)
                    draw=ImageDraw.Draw(imgok)
                    draw.text((5,5),'recording',fill=(255,0,0),font=font)
                    self.display.ShowImage(imgok)
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
                    if XGOEDU.xgoButton(self,"b"):
                        break
                time.sleep(1)
                videoWrite.release()
            if XGOEDU.xgoButton(self,"c"):
                XGOEDU.lcd_clear(self)
                time.sleep(0.5)
                break
    '''
    骨骼识别
    '''
    def posenetRecognition(self,target="camera"):
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        ges = ''
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        joint_list = [[24,26,28], [23,25,27], [14,12,24], [13,11,23]]  # leg&arm
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            image=np.array(Image.open(target))

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                
                if results.pose_landmarks:
                    RHL = results.pose_landmarks
                    angellist=[]
                    for joint in joint_list:
                        a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
                        b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
                        c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
                        radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                        angle = np.abs(radians_fingers * 180.0 / np.pi) 
                        if angle > 180.0:
                            angle = 360 - angle
                        #cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        angellist.append(angle)
                else:
                    angellist=[]
                print(angellist)
                b,g,r = cv2.split(image)
                image = cv2.merge((r,g,b))
                image = cv2.flip(image, 1)
                try:
                    ges=str(int(angellist[0]))+'|'+str(int(angellist[1]))+'|'+str(int(angellist[2]))+'|'+str(int(angellist[3]))
                except:
                    ges=' '
                cv2.putText(image,ges,(10,220),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                imgok = Image.fromarray(image)
                self.display.ShowImage(imgok)


        # datas = self.hand.run(image)
        # b,g,r = cv2.split(image)
        # image = cv2.merge((r,g,b))
        # #image = cv2.flip(image,1)
        # for data in datas:
        #     rect = data['rect']
        #     right_left = data['right_left']
        #     center = data['center']
        #     dlandmark = data['dlandmark']
        #     hand_angle = data['hand_angle']
        #     XGOEDU.rectangle(self,image,rect,"#33cc00",2)
        #     #XGOEDU.text(self,image,right_left,center,2,"#cc0000",5)
        #     if right_left == 'L':
        #         XGOEDU.text(self,image,hand_pos(hand_angle),(180,80),1.5,"#33cc00",2)
        #     elif right_left == 'R':
        #         XGOEDU.text(self,image,hand_pos(hand_angle),(50,80),1.5,"#ff0000",2)
        #     ges = hand_pos(hand_angle)
        #     for i in dlandmark:
        #         XGOEDU.circle(self,image,i,3,"#ff9900",-1)
        # imgok = Image.fromarray(image)
        # self.display.ShowImage(imgok)
        if angellist==[]:
            return None
        else:
            return angellist

    '''
    手势识别
    '''
    def gestureRecognition(self,target="camera"):
        ges = ''
        if self.hand==None:
            self.hand = hands(0,2,0.6,0.5)
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            image=np.array(Image.open(target))
        image = cv2.flip(image,1)
        datas = self.hand.run(image)
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        for data in datas:
            rect = data['rect']
            right_left = data['right_left']
            center = data['center']
            dlandmark = data['dlandmark']
            hand_angle = data['hand_angle']
            XGOEDU.rectangle(self,image,rect,"#33cc00",2)
            #XGOEDU.text(self,image,right_left,center,2,"#cc0000",5)
            if right_left == 'L':
                XGOEDU.text(self,image,hand_pos(hand_angle),(180,80),1.5,"#33cc00",2)
            elif right_left == 'R':
                XGOEDU.text(self,image,hand_pos(hand_angle),(50,80),1.5,"#ff0000",2)
            ges = hand_pos(hand_angle)
            for i in dlandmark:
                XGOEDU.circle(self,image,i,3,"#ff9900",-1)
        imgok = Image.fromarray(image)
        self.display.ShowImage(imgok)
        if ges=='':
            return None
        else:
            return(ges,center)
    '''
    yolo
    '''
    def yoloFast(self,target="camera"):
        ret=''
        self.open_camera()
        if self.yolo==None:
            self.yolo = yoloXgo('/home/pi/model/Model.onnx',
            ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'],
            [352,352],0.66)
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            image=np.array(Image.open(target))
        datas = self.yolo.run(image)
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        image = cv2.flip(image,1)
        if datas:
            for data in datas:
                XGOEDU.rectangle(self,image,data['xywh'],"#33cc00",2)
                xy= (data['xywh'][0], data['xywh'][1])
                XGOEDU.text(self,image,data['classes'],xy,1,"#ff0000",2)
                value_yolo = data['classes']
                ret=(value_yolo,xy)
        imgok = Image.fromarray(image)
        self.display.ShowImage(imgok)
        if ret=='':
            return None
        else:
            return ret

    '''
    人脸坐标点检测
    '''
    def face_detect(self,target="camera"):
        ret=''
        if self.face==None:
            self.face = face_detection(0.7)
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            image=np.array(Image.open(target))
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        image = cv2.flip(image,1)
        datas = self.face.run(image)
        for data in datas:
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
            ret=data['rect']
        imgok = Image.fromarray(image)
        self.display.ShowImage(imgok)
        if ret=='':
            return None
        else:
            return ret

    '''
    情绪识别
    '''
    def emotion(self,target="camera"):
        ret=''
        if self.classifier==None:
            from keras.models import load_model
            self.face_classifier=cv2.CascadeClassifier('/home/pi/model/haarcascade_frontalface_default.xml')
            self.classifier = load_model('/home/pi/model/EmotionDetectionModel.h5')
        class_labels=['Angry','Happy','Neutral','Sad','Surprise']
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            image=np.array(Image.open(target))
        labels=[]
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces=self.face_classifier.detectMultiScale(gray,1.3,5)
        label=''
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                from tensorflow.keras.utils import img_to_array
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)

                preds=self.classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                ret=(label,(x,y))
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
        self.display.ShowImage(imgok)
        if ret=='':
            return None
        else:
            return ret

    '''
    年纪及性别检测
    '''
    def agesex(self,target="camera"):
        ret=''
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']
        padding = 20
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            image=np.array(Image.open(target))
        if self.agesexmark==None:
            faceProto = "/home/pi/model/opencv_face_detector.pbtxt"
            faceModel = "/home/pi/model/opencv_face_detector_uint8.pb"
            ageProto = "/home/pi/model/age_deploy.prototxt"
            ageModel = "/home/pi/model/age_net.caffemodel"
            genderProto = "/home/pi/model/gender_deploy.prototxt"
            genderModel = "/home/pi/model/gender_net.caffemodel"
            self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
            self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
            self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
            self.agesexmark=True

        image = cv2.flip(image, 1)
        frameFace, bboxes = getFaceBox(self.faceNet, image)
        gender=''
        age=''
        for bbox in bboxes:
            face = image[max(0, bbox[1] - padding):min(bbox[3] + padding, image.shape[0] - 1),
                    max(0, bbox[0] - padding):min(bbox[2] + padding, image.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)   
            genderPreds = self.genderNet.forward()   
            gender = genderList[genderPreds[0].argmax()]  
            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,cv2.LINE_AA)  
            ret=(gender,age,(bbox[0], bbox[1]))
        b,g,r = cv2.split(frameFace)
        frameFace = cv2.merge((r,g,b))
        imgok = Image.fromarray(frameFace)
        self.display.ShowImage(imgok)
        if ret=='':
            return None
        else:
            return ret

    
    def rectangle(self,frame,z,colors,size):
        frame=cv2.rectangle(frame,(int(z[0]),int(z[1])),(int(z[0]+z[2]),int(z[1]+z[3])),color(colors),size)
        return frame
        
    def circle(self,frame,xy,rad,colors,tk):
        frame=cv2.circle(frame,xy,rad,color(colors),tk)
        return frame
    
    def text(self,frame,text,xy,font_size,colors,size):
        frame=cv2.putText(frame,text,xy,cv2.FONT_HERSHEY_SIMPLEX,font_size,color(colors),size)
        return frame   

    def SpeechRecognition(self,seconds=3):
        self.xgoAudioRecord(filename="recog",seconds=seconds)
        from urllib.request import urlopen
        from urllib.request import Request
        from urllib.error import URLError
        from urllib.parse import urlencode
        timer = time.perf_counter
        AUDIO_FILE = 'recog.wav' 
        FORMAT = AUDIO_FILE[-3:]  
        CUID = '123456PYTHON'
        RATE = 16000
        DEV_PID = 1537  
        ASR_URL = 'http://vop.baidu.com/server_api'
        SCOPE = 'audio_voice_assistant_get' 

        token = self.fetch_token()

        speech_data = []
        path="/home/pi/xgoMusic/"
        with open(path+AUDIO_FILE, 'rb') as speech_file:
            speech_data = speech_file.read()

        length = len(speech_data)
        if length == 0:
            raise DemoError('file %s length read 0 bytes' % AUDIO_FILE)
        speech = base64.b64encode(speech_data)
        speech = str(speech, 'utf-8')
        params = {'dev_pid': DEV_PID,
                'format': FORMAT,
                'rate': RATE,
                'token': token,
                'cuid': CUID,
                'channel': 1,
                'speech': speech,
                'len': length
                }
        post_data = json.dumps(params, sort_keys=False)
        req = Request(ASR_URL, post_data.encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        try:
            begin = timer()
            f = urlopen(req)
            result_str = f.read()
            print ("Request time cost %f" % (timer() - begin))
        except URLError as err:
            print('asr http response http code : ' + str(err.code))
            result_str = err.read()
        try:
            result_str = str(result_str, 'utf-8')
            re=json.loads(result_str)
            text=re['result'][0]
        except:
            text='error!'
        return text

    def SpeechSynthesis(self,texts):
        from urllib.request import urlopen
        from urllib.request import Request
        from urllib.error import URLError
        from urllib.parse import urlencode
        from urllib.parse import quote_plus

        TEXT = texts
        PER = 0
        SPD = 5
        PIT = 5
        VOL = 5
        AUE = 6
        FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
        FORMAT = FORMATS[AUE]
        CUID = "123456PYTHON"
        TTS_URL = 'http://tsn.baidu.com/text2audio'

        SCOPE = 'audio_tts_post' 

        token = self.fetch_token()
        tex = quote_plus(TEXT) 
        print(tex)
        params = {'tok': token, 'tex': tex, 'per': PER, 'spd': SPD, 'pit': PIT, 'vol': VOL, 'aue': AUE, 'cuid': CUID,
                'lan': 'zh', 'ctp': 1}  

        data = urlencode(params)
        print('test on Web Browser' + TTS_URL + '?' + data)

        req = Request(TTS_URL, data.encode('utf-8'))
        has_error = False
        try:
            f = urlopen(req)
            result_str = f.read()

            headers = dict((name.lower(), value) for name, value in f.headers.items())

            has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
        except  URLError as err:
            print('asr http response http code : ' + str(err.code))
            result_str = err.read()
            has_error = True

        path="/home/pi/xgoMusic/"
        save_file = "error.txt" if has_error else 'result.' + FORMAT
        with open(path+save_file, 'wb') as of:
            of.write(result_str)

        if has_error:
            result_str = str(result_str, 'utf-8')
            print("tts api  error:" + result_str)

        print("result saved as :" + save_file)

        self.xgoSpeaker("result.wav")

    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontStyle = ImageFont.truetype(
            "/home/pi/model/msyh.ttc", textSize, encoding="utf-8")
        draw.text(position, text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
    def QRRecognition(self,target="camera"):
        import pyzbar.pyzbar as pyzbar
        if target=="camera":
            self.open_camera()
            success,img = self.cap.read()
        else:
            path="/home/pi/xgoPictures/"
            img=np.array(Image.open(path+target))
     
        barcodes = pyzbar.decode(img) 
        result=[]
        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            result.append(barcodeData)
            text = "{} ({})".format(barcodeData, barcodeType)
            img=self.cv2AddChineseText(img,text, (10, 30),(0, 255, 0), 30)
        try:
            re=result[0]
        except:
            result=[]
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        imgok = Image.fromarray(img)
        self.display.ShowImage(imgok)
        return result

    def ColorRecognition(self,target="camera",mode='R'):
        color_x = 0
        color_y = 0
        color_radius = 0

        if mode=='R':  #red
            color_lower = np.array([0, 43, 46])
            color_upper = np.array([10, 255, 255])
        elif mode=='G': #green
            color_lower = np.array([35, 43, 46])
            color_upper = np.array([77, 255, 255])
        elif mode=='B':   #blue
            color_lower = np.array([100, 43, 46])
            color_upper = np.array([124, 255, 255])
        elif mode=='Y':   #yellow
            color_lower = np.array([26, 43, 46])
            color_upper = np.array([34, 255, 255])
        if target=="camera":
            self.open_camera()
            success,frame = self.cap.read()
        else:
            path="/home/pi/xgoPictures/"
            frame=np.array(Image.open(path+target))
        frame_ = cv2.GaussianBlur(frame,(5,5),0)                    
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,color_lower,color_upper)  
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        mask = cv2.GaussianBlur(mask,(3,3),0)     
        cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] 

        if len(cnts) > 0:
            cnt = max (cnts, key = cv2.contourArea)
            (color_x,color_y),color_radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame,(int(color_x),int(color_y)),int(color_radius),(255,0,255),2)  
        cv2.putText(frame, "X:%d, Y%d" % (int(color_x), int(color_y)), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 3)

        b,g,r = cv2.split(frame)
        img = cv2.merge((r,g,b))
        imgok = Image.fromarray(img)
        self.display.ShowImage(imgok)

        return ((color_x,color_y),color_radius)

    def cap_color_mask(self,position=None, scale=25, h_error=20, s_limit=[90, 255], v_limit=[90, 230]):
        if position is None:
            position = [160, 100]
        count = 0
        self.open_camera()
        while True:
            if self.xgoButton("c"):   
                break
            success,frame = self.cap.read()
            b,g,r = cv2.split(frame)
            frame_bgr = cv2.merge((r,g,b))
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            color = np.mean(h[position[1]:position[1] + scale, position[0]:position[0] + scale])
            if self.xgoButton("b") and count == 0:
                count += 1
                color = np.mean(h[position[1]:position[1] + scale, position[0]:position[0] + scale])
                color_lower = [max(color - h_error, 0), s_limit[0], v_limit[0]]
                color_upper = [min(color + h_error, 255), s_limit[1], v_limit[1]]
                return [color_lower, color_upper]

            if count == 0:
                cv2.rectangle(frame, (position[0], position[1]), (position[0] + scale, position[1] + scale),
                            (255, 255, 255), 2)
                cv2.putText(frame, 'press button B', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            b,g,r = cv2.split(frame)
            img = cv2.merge((r,g,b))
            imgok = Image.fromarray(img)
            self.display.ShowImage(imgok)
    
    def filter_img(self,frame,color):
        b,g,r = cv2.split(frame)
        frame_bgr = cv2.merge((r,g,b))
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if isinstance(color, list):
            color_lower = np.array(color[0])
            color_upper = np.array(color[1])
        else:
            color_upper, color_lower = get_color_mask(color)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        img_mask = cv2.bitwise_and(frame, frame, mask=mask)
        return img_mask

    def BallRecognition(self,color_mask,target="camera",p1=36, p2=15, minR=6, maxR=35):
        x=y=ra=0
        if target=="camera":
            self.open_camera()
            success,image = self.cap.read()
        else:
            path="/home/pi/xgoPictures/"
            image=np.array(Image.open(path+target))

        frame_mask=self.filter_img(image, color_mask)
        
        img = cv2.medianBlur(frame_mask, 5)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=minR,maxRadius=maxR)
        b,g,r = cv2.split(image)
        image = cv2.merge((r,g,b))
        if circles is not None and len(circles[0]) == 1:
            param = circles[0][0]
            x, y, ra = int(param[0]), int(param[1]), int(param[2])
            cv2.circle(image, (x, y), ra, (255, 255, 255), 2)
            cv2.circle(image, (x, y), 2, (255, 255, 255), 2)
        imgok = Image.fromarray(image)
        self.display.ShowImage(imgok)
        return x,y,ra





class DemoError(Exception):
    pass

class hands():
    def __init__(self,model_complexity,max_num_hands,min_detection_confidence,min_tracking_confidence):
        import mediapipe as mp
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
        import copy
        image = cv_img
        debug_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        M = cv2.moments(palm_array)
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
        x, y, w, h = cv2.boundingRect(landmark_array)
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
        import onnxruntime 
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
        import mediapipe as mp
        self.model_selection = 0
        self.min_detection_confidence =min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
        )

    def run(self,cv_img):
        image = cv_img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
