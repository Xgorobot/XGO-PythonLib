# XGO-PythonLib

XGO2 robot can be developed using the Python language.The motion,pose,coordinates,gripper and servo of the XGO2 can be controlled via Python. The detailed instructions for use of the library are as follows.

PythonLib included xgolib.py xgoedu.py xgoadvance.py.

[Luwu Dynamics · WIKI](https://www.yuque.com/luwudynamics)

[PythonLib-WIKI](https://www.yuque.com/luwudynamics/cn/mxkaodwpo2h5zmvw)



## Install instructions 

1 Burn the official 0512 img image 

2 Copy all files from the "model" directory to `\home\pi\model`

3 Run this command:

```
pip install --upgrade xgo-pythonlib
```

## Examples

Perform gesture recognition on the current camera and press the "c" key to exit.

```python
from xgoedu import XGOEDU 
edu = XGOEDU()

while True:
    result=edu.gestureRecognition()  
    print(result)
    if edu.xgoButton("c"):  
        break
```
xgolib library example
```python
from xgolib import XGO
dog = XGO('/dev/ttyAMA0')
dog.action(1)
```
### Lastest Verion:0.1.3

### xgolib_version=1.3.0

### xgoedu_version=1.2.1



