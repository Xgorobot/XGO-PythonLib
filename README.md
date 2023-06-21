# XGO-PythonLib

XGO2 has built-in motion libraries for controlling the movement and various features of the machine dog, including battery level, firmware version, and servo angle. The motion library enables users to control translation and pose movement, as well as single servo and single-leg movement. The education library facilitates camera, screen, key, microphone, and speaker operations, as well as commonly used AI functions such as gesture recognition, face detection, emotional recognition, and age and gender recognition.  The detailed instructions for use of the library are as follows.

PythonLib included xgolib.py and xgoedu.py

[Luwu Dynamics · WIKI](https://www.yuque.com/luwudynamics)

[PythonLib-WIKI](https://www.yuque.com/luwudynamics/cn/mxkaodwpo2h5zmvw)



## Install instructions 

1 Burn the official 0609 img image 

2 Copy all files from the "model" directory to `\home\pi\model`

3 Run this command:

```
pip install --upgrade xgo-pythonlib
sudo pip install --upgrade xgo-pythonlib
```

## Examples

Perform gesture recognition on the current camera and press the "c" key to exit.

```python
from xgoedu import XGOEDU 
XGO_edu = XGOEDU()

while True:
    result=XGO_edu.gestureRecognition()  
    print(result)
    if XGO_edu.xgoButton("c"):  
        break
```
xgolib library example
```python
from xgolib import XGO
dog = XGO('xgomini')
dog.action(1)
```
### 

## Change Log

### [0.2.0] - 2023-06-21

#### Fixed

- xgoVideo and xgoVideoRecord method can be used.

### [0.1.9] - 2023-06-20

#### Fixed

- Fixed the issue with the xgoTakePhoto method that was causing abnormal RGB colors in the saved photos.



