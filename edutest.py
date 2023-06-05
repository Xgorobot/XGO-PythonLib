from xgoedu import XGOEDU
dog=XGOEDU()
import time

# ps -ef | grep main.py

# dog.lcd_line(10,10,100,100,(255,200,45),5)
# dog.lcd_line(10,10,100,100)
# dog.lcd_circle(10,10,100,100,10,100,(55,200,200),5)
# dog.lcd_rectangle(10,10,100,100,fill=(255,0,0),outline=(0,255,0),width=5)
# dog.lcd_text(30,30,'中文测试',color=(255,0,0),fontsize=40)
# dog.lcd_picture("test.jpg",30,30)

# while 1:
#     print(dog.xgoButton("a"))
#     time.sleep(0.1)

# dog.xgoSpeaker("test.mp3")

# dog.xgoAudioRecord("test.wav",5)

# dog.cameraOn()

# while 1:
#     print(dog.gestureRecognition())

# while 1:
#     print(dog.yoloFast("camera.jpg"))
    
# while 1:
#     print(dog.face_detect("camera.jpg"))

while 1:
    print(dog.emotion("camera.jpg"))

# while 1:
#     print(dog.agesex("camera.jpg"))



