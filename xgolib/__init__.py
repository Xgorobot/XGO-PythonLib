import serial
import struct
import time
import math

__version__ = '1.5'
__last_modified__ = '2024/12/18'

"""
XGOorder is used to store the command address and corresponding data.
"""

XGOorder = {
    "BATTERY": [0x01, 100],
    "PERFORM": [0x03, 0],
    "CALIBRATION": [0x04, 0],
    "UPGRADE": [0x05, 0],
    "SET_ORIGIN": [0x06, 1],
    "FIRMWARE_VERSION": [0x07],
    "GAIT_TYPE": [0x09, 0x00],
    "BT_NAME": [0x13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "UNLOAD_MOTOR": [0x20, 0],
    "LOAD_MOTOR": [0x20, 0],
    "VX": [0x30, 128],
    "VY": [0x31, 128],
    "VYAW": [0x32, 128],
    "TRANSLATION": [0x33, 0, 0, 0],
    "ATTITUDE": [0x36, 0, 0, 0],
    "PERIODIC_ROT": [0x39, 0, 0, 0],
    "MarkTime": [0x3C, 0],
    "MOVE_MODE": [0x3D, 0],
    "ACTION": [0x3E, 0],
    "MOVE_TO": [0x3F, 0, 0],
    "PERIODIC_TRAN": [0x80, 0, 0, 0],
    "MOTOR_ANGLE": [0x50, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
    "MOTOR_SPEED": [0x5C, 1],
    "MOVE_TO_MID": [0x5F, 1],
    "LEG_POS": [0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IMU": [0x61, 0],
    "ROLL": [0x62, 0],
    "PITCH": [0x63, 0],
    "TEACH_RECORD": [0x21, 0],
    "TEACH_PLAY": [0x22, 0],
    "TEACH_ARM_RECORD": [0x23, 0],
    "TEACH_ARM_PLAY": [0x24, 0],
    "YAW": [0x64, 0],
    "CLAW": [0x71, 0],
    "ARM_MODE": [0x72, 0],
    "ARM_X": [0x73, 0],
    "ARM_Z": [0x74, 0],
    "ARM_SPEED": [0x75, 0],
    "ARM_THETA": [0x76, 0],
    "ARM_R": [0x77, 0],
    "OUTPUT_ANALOG": [0x90, 0],
    "OUTPUT_DIGITAL": [0x91, 0],
    "LED_COLOR": [0x69, 0, 0, 0]
}

"""
XGOparam is used to store the parameter limit range of the robot dog.
"""
XGOparam = {}

def search(data, list):
    """
    Searches for a specific data element within a list.

    Parameters:
        data: The data element to search for.
        list: The list to search within.

    Returns:
        int: The index (position + 1) of the data element in the list if found, otherwise -1.
    """
    for i in range(len(list)):
        if data == list[i]:
            return i + 1
    return -1

def conver2u8(data, limit, min_value=0):
    """
    Converts the actual parameters to single byte data from 0 to 255.

    Parameters:
        data (float): The actual parameter value.
        limit (float or list): The parameter limit range. If a single float is provided, it's treated as the maximum limit with a symmetrical negative limit. If a list is provided, it should contain two elements: [min_limit, max_limit].
        min_value (int, optional): The minimum value to return if the data is below the limit. Defaults to 0.

    Returns:
        int: The converted single byte data (0-255).
    """
    max_value = 0xff
    if not isinstance(limit, list):
        limit = [-limit, limit]
    if data >= limit[1]:
        return max_value
    elif data <= limit[0]:
        return min_value
    else:
        return int(255 / (limit[1] - limit[0]) * (data - limit[0]))

def conver2float(data, limit):
    """
    Converts a single byte data (0-255) to its corresponding float value based on the provided limit.

    Parameters:
        data (int): The single byte data (0-255).
        limit (float or list): The parameter limit range. If a single float is provided, it's treated as the maximum limit with a symmetrical negative limit. If a list is provided, it should contain two elements: [min_limit, max_limit].

    Returns:
        float: The corresponding float value.
    """
    if not isinstance(limit, list):
        return (data - 128.0) / 255.0 * limit
    else:
        return data / 255.0 * (limit[1] - limit[0]) + limit[0]

def Byte2Float(rawdata):
    """
    Converts a 4-byte sequence (in little-endian format) to a float.

    Parameters:
        rawdata (list): A list containing 4 bytes.

    Returns:
        float: The converted float value.
    """
    a = bytearray()
    a.append(rawdata[3])
    a.append(rawdata[2])
    a.append(rawdata[1])
    a.append(rawdata[0])
    return struct.unpack("!f", a)[0]

def Byte2Short(rawdata):
    """
    Converts a 2-byte sequence (in big-endian format) to a signed short integer.

    Parameters:
        rawdata (list): A list containing 2 bytes.

    Returns:
        int: The converted signed short integer value.
    """
    a = bytearray()
    a.append(rawdata[0])
    a.append(rawdata[1])
    return struct.unpack('>h', a)[0]

def changePara(version):
    """
    Changes the XGOparam dictionary based on the robot version.

    Parameters:
        version (str): The robot version ('xgomini', 'xgolite', or 'xgorider').
    """
    global XGOparam
    if version == 'xgomini':
        XGOparam = {
            "TRANSLATION_LIMIT": [35, 19.5, [75, 120]],  # X Y Z translation range
            "ATTITUDE_LIMIT": [20, 22, 16],  # Roll Pitch Yaw attitude range
            "LEG_LIMIT": [35, 18, [75, 115]],  # Leg length range
            "MOTOR_LIMIT": [[-73, 57], [-66, 93], [-31, 31], [-65, 65], [-85, 50], [-75, 90]],  # Lower, middle, upper servo range
            "PERIOD_LIMIT": [[1.5, 8]],
            "MARK_TIME_LIMIT": [10, 35],  # Mark time height range
            "VX_LIMIT": 25,  # X speed range
            "VY_LIMIT": 18,  # Y speed range
            "VYAW_LIMIT": 100,  # Rotation speed range
            "ARM_LIMIT": [[-80, 155], [-95, 155], [70, 270], [80, 140]],
            "ActionTime": {
                1: 3, 2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 7,
                11: 7, 12: 5, 13: 7, 14: 10, 15: 6, 16: 6, 17: 4, 18: 6, 19: 10, 20: 9,
                21: 8, 22: 7, 23: 6, 24: 7, 128: 10, 129: 10, 130: 10, 255: 1}
        }
    elif version == 'xgolite':
        XGOparam = {
            "TRANSLATION_LIMIT": [25, 18, [60, 110]],
            "ATTITUDE_LIMIT": [20, 10, 12],
            "LEG_LIMIT": [25, 18, [60, 110]],
            "MOTOR_LIMIT": [[-70, 50], [-70, 90], [-30, 30], [-65, 65], [-115, 70], [-85, 100]],
            "PERIOD_LIMIT": [[1.5, 8]],
            "MARK_TIME_LIMIT": [10, 25],
            "VX_LIMIT": 25,
            "VY_LIMIT": 18,
            "VYAW_LIMIT": 100,
            "ARM_LIMIT": [[-80, 155], [-95, 155], [70, 270], [80, 140]],
            "ActionTime": {
                1: 3, 2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 7,
                11: 7, 12: 5, 13: 7, 14: 10, 15: 6, 16: 6, 17: 4, 18: 6, 19: 10, 20: 9,
                21: 8, 22: 7, 23: 6, 24: 7, 128: 10, 129: 10, 130: 10, 255: 1}
        }
    elif version == "xgorider":
        XGOparam = {
            "TRANSLATION_LIMIT": [1, 1, [60, 120]],
            "ATTITUDE_LIMIT": [17, 1, 1],
            "LEG_LIMIT": [1, 1, [60, 120]],
            "MOTOR_LIMIT": [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]],
            "PERIOD_LIMIT": [[1, 2]],
            "MARK_TIME_LIMIT": [-1, 1],
            "VX_LIMIT": 1.5,
            "VY_LIMIT": 1.0,
            "VYAW_LIMIT": 360,
            "ARM_LIMIT": [[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
            "ActionTime": {
                1: 3, 2: 3, 3: 5, 4: 5, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 7,
                11: 7, 12: 5, 13: 7, 14: 10, 15: 6, 16: 6, 17: 4, 18: 6, 19: 10, 20: 9,
                21: 8, 22: 7, 23: 6, 24: 7, 128: 10, 129: 10, 130: 10, 255: 1}
        }

class XGO():
    """
    When instantiating XGO, you need to specify the serial communication interface between the upper computer and the machine dog.
    """

    def __init__(self, port, baud=115200, version="xgomini", verbose=False):
        """
        Initializes the XGO robot object.

        Parameters:
            port (str): The serial port to use for communication (e.g., '/dev/ttyACM0', 'COM3').
            baud (int, optional): The baud rate for serial communication. Defaults to 115200.
            version (string, optional): Specifies the version of the XGO robot. Accepts 'xgomini', 'xgolite', or 'xgorider'. Defaults to 'xgomini'.
            verbose (bool, optional): Enables verbose output for debugging. Defaults to False.
        """
        self.verbose = verbose
        self.ser = serial.Serial("/dev/ttyAMA0", baud, timeout=0.5)
        self.ser.flushOutput()
        self.ser.flushInput()
        self.port = port
        self.rx_FLAG = 0
        self.rx_COUNT = 0
        self.rx_ADDR = 0
        self.rx_LEN = 0
        self.rx_data = bytearray(50)
        time.sleep(0.25)
        self.version = self.read_firmware()
        if self.version[0] == 'M':
            changePara('xgomini')
        elif self.version[0] == 'L':
            changePara('xgolite')
        elif self.version[0] == 'R':
            changePara('xgorider')
        else:
            changePara('xgomini')
            print("ERROR!Can't read firmware version!")
        self.mintime = 0.65
        self.reset()
        self.init_yaw = self.read_yaw()
        time.sleep(1)
        pass

    def __send(self, key, index=1, len=1):
        """
        Sends a command to the XGO robot.

        Parameters:
            key (str): The command key (e.g., "VX", "VY", "TRANSLATION").
            index (int, optional): The starting index for the data within the XGOorder dictionary. Defaults to 1.
            len (int, optional): The number of data elements to send. Defaults to 1.
        """
        mode = 0x01
        order = XGOorder[key][0] + index - 1
        value = []
        value_sum = 0
        for i in range(0, len):
            value.append(XGOorder[key][index + i])
            value_sum = value_sum + XGOorder[key][index + i]
        sum_data = ((len + 0x08) + mode + order + value_sum) % 256
        sum_data = 255 - sum_data
        tx = [0x55, 0x00, (len + 0x08), mode, order]
        tx.extend(value)
        tx.extend([sum_data, 0x00, 0xAA])
        self.ser.write(tx)
        if self.verbose:
            print("tx_data: ", tx)

    def __read(self, addr, read_len=1):
        """
        Sends a read request to the XGO robot.

        Parameters:
            addr (int): The address to read from.
            read_len (int, optional): The number of bytes to read. Defaults to 1.
        """
        self.ser.flushInput()
        mode = 0x02
        sum_data = (0x09 + mode + addr + read_len) % 256
        sum_data = 255 - sum_data
        tx = [0x55, 0x00, 0x09, mode, addr, read_len, sum_data, 0x00, 0xAA]
        self.ser.flushInput()
        self.ser.write(tx)
        if self.verbose:
            print("tx_data: ", tx)

    def __change_baud(self, baud):
        """
        Changes the baud rate of the serial connection.

        Parameters:
            baud (int): The new baud rate.
        """
        self.ser.flush()
        self.ser.close()
        self.ser = serial.Serial(self.port, baud, timeout=0.5)

    def stop(self):
        """
        Stops all movement of the XGO robot.
        """
        self.move_x(0)
        self.move_y(0)
        self.mark_time(0)
        self.turn(0)

    def move(self, direction, step):
        """
        Moves the XGO robot in a specified direction.

        Parameters:
            direction (str): The direction of movement ('x', 'X', 'y', or 'Y').
            step (float): The step size or speed of the movement.

        Raises:
            ValueError: If an invalid direction is provided.
        """
        if direction in ['x', 'X']:
            self.move_x(step)
        elif direction in ['y', 'Y']:
            self.move_y(step)
        else:
            print("ERROR!Invalid direction!")

    def move_x(self, step, runtime=0):
        """
        Moves the XGO robot along the x-axis.

        Parameters:
            step (float): The step size or speed of the movement along the x-axis.
            runtime (float, optional): The duration of the movement in seconds. If provided, the robot will stop moving after this duration. Defaults to 0.
        """
        XGOorder["VX"][1] = conver2u8(step, XGOparam["VX_LIMIT"])
        self.__send("VX")
        if runtime:
            time.sleep(runtime)
            XGOorder["VX"][1] = conver2u8(0, XGOparam["VX_LIMIT"])
            self.__send("VX")

    def move_y(self, step, runtime=0):
        """
        Moves the XGO robot along the y-axis.

        Parameters:
            step (float): The step size or speed of the movement along the y-axis.
            runtime (float, optional): The duration of the movement in seconds. If provided, the robot will stop moving after this duration. Defaults to 0.
        """
        XGOorder["VY"][1] = conver2u8(step, XGOparam["VY_LIMIT"])
        self.__send("VY")
        if runtime:
            time.sleep(runtime)
            XGOorder["VY"][1] = conver2u8(0, XGOparam["VY_LIMIT"])
            self.__send("VY")

    def turn(self, step, runtime=0):
        """
        Rotates the XGO robot.

        Parameters:
            step (float): The step size or speed of the rotation.
            runtime (float, optional): The duration of the rotation in seconds. If provided, the robot will stop rotating after this duration. Defaults to 0.
        """
        XGOorder["VYAW"][1] = conver2u8(step, XGOparam["VYAW_LIMIT"])
        self.__send("VYAW")
        if runtime:
            time.sleep(runtime)
            XGOorder["VYAW"][1] = conver2u8(0, XGOparam["VYAW_LIMIT"])
            self.__send("VYAW")

    def forward(self, step):
        """
        Moves the XGO robot forward.

        Parameters:
            step (float): The step size or speed of the forward movement.
        """
        self.move_x(abs(step))

    def back(self, step):
        """
        Moves the XGO robot backward.

        Parameters:
            step (float): The step size or speed of the backward movement.
        """
        self.move_x(-abs(step))

    def left(self, step):
        """
        Moves the XGO robot to the left.

        Parameters:
            step (float): The step size or speed of the leftward movement.
        """
        self.move_y(abs(step))

    def right(self, step):
        """
        Moves the XGO robot to the right.

        Parameters:
            step (float): The step size or speed of the rightward movement.
        """
        self.move_y(-abs(step))

    def turnleft(self, step):
        """
        Turns the XGO robot to the left.

        Parameters:
            step (float): The step size or speed of the left turn.
        """
        self.turn(abs(step))

    def turnright(self, step):
        """
        Turns the XGO robot to the right.

        Parameters:
            step (float): The step size or speed of the right turn.
        """
        self.turn(-abs(step))

    def move_by(self, distance, vx, vy, k, mintime):
        """
        Moves the XGO robot a specific distance using a combination of x and y velocities.

        Parameters:
            distance (float): The distance to move.
            vx (float): The velocity along the x-axis.
            vy (float): The velocity along the y-axis.
            k (float): A scaling factor for the movement duration.
            mintime (float): The minimum duration of the movement.
        """
        runtime = k * abs(distance) + mintime
        self.move_x(math.copysign(vx, distance))
        self.move_y(math.copysign(vy, distance))
        time.sleep(runtime)
        self.move_x(0)
        self.move_y(0)
        time.sleep(0.2)

    def move_x_by(self, distance, vx=18, k=0.035, mintime=0.55):
        """
        Moves the XGO robot a specific distance along the x-axis.

        Parameters:
            distance (float): The distance to move along the x-axis.
            vx (float, optional): The velocity along the x-axis. Defaults to 18.
            k (float, optional): A scaling factor for the movement duration. Defaults to 0.035.
            mintime (float, optional): The minimum duration of the movement. Defaults to 0.55.
        """
        self.move_by(distance, vx, 0, k, mintime)
        pass

    def move_y_by(self, distance, vy=18, k=0.0373, mintime=0.5):
        """
        Moves the XGO robot a specific distance along the y-axis.

        Parameters:
            distance (float): The distance to move along the y-axis.
            vy (float, optional): The velocity along the y-axis. Defaults to 18.
            k (float, optional): A scaling factor for the movement duration. Defaults to 0.0373.
            mintime (float, optional): The minimum duration of the movement. Defaults to 0.5.
        """
        self.move_by(distance, 0, vy, k, mintime)
        pass

    def turn_by(self, theta, mintime, vyaw=16, k=0.08):
        """
        Turns the XGO robot by a specific angle.

        Parameters:
            theta (float): The angle to turn (in degrees).
            mintime (float): The minimum duration of the turn.
            vyaw (float, optional): The angular velocity. Defaults to 16.
            k (float, optional): A scaling factor for the turn duration. Defaults to 0.08.
        """
        runtime = abs(theta) * k + mintime
        self.turn(math.copysign(vyaw, theta))
        time.sleep(runtime)
        self.turn(0)
        pass

    def turn_to(self, theta, vyaw=60, emax=10):
        """
        Turns the XGO robot to a specific absolute angle.

        Parameters:
            theta (float): The target angle (in degrees).
            vyaw (float, optional): The angular velocity. Defaults to 60.
            emax (float, optional): The maximum error tolerance for the angle. Defaults to 10.
        """
        cur_yaw = self.read_yaw()
        des_yaw = self.init_yaw + theta
        while abs(des_yaw - cur_yaw) >= emax:
            self.turn(math.copysign(vyaw, des_yaw - cur_yaw))
            cur_yaw = self.read_yaw()
        self.turn(0)
        time.sleep(0.2)
        pass

    def __translation(self, direction, data):
        """
        Translates the XGO robot's body along a specified axis.

        Parameters:
            direction (str): The axis of translation ('x', 'y', or 'z').
            data (float): The translation amount.
        """
        index = search(direction, ['x', 'y', 'z'])
        if index == -1:
            print("ERROR!Direction must be 'x', 'y' or 'z'")
            return
        XGOorder["TRANSLATION"][index] = conver2u8(data, XGOparam["TRANSLATION_LIMIT"][index - 1])
        self.__send("TRANSLATION", index)

    def translation(self, direction, data):
        """
        Translates the XGO robot's body.

        Parameters:
            direction (str or list): The axis/axes of translation ('x', 'y', 'z', or a list of these).
            data (float or list): The translation amount(s).

        Raises:
            ValueError: If the length of direction and data don't match when using a list.
        """
        if isinstance(direction, list):
            if len(direction) != len(data):
                print("ERROR!The length of direction and data don't match!")
                return
            for i in range(len(data)):
                self.__translation(direction[i], data[i])
        else:
            self.__translation(direction, data)

    def __attitude(self, direction, data):
        """
        Adjusts the XGO robot's body attitude along a specified axis.

        Parameters:
            direction (str): The axis of attitude adjustment ('r' for roll, 'p' for pitch, or 'y' for yaw).
            data (float): The attitude adjustment amount.
        """
        index = search(direction, ['r', 'p', 'y'])
        if index == -1:
            print("ERROR!Direction must be 'r', 'p' or 'y'")
            return
        XGOorder["ATTITUDE"][index] = conver2u8(data, XGOparam["ATTITUDE_LIMIT"][index - 1])
        self.__send("ATTITUDE", index)

    def attitude(self, direction, data):
        """
        Adjusts the XGO robot's body attitude.

        Parameters:
            direction (str or list): The axis/axes of attitude adjustment ('r', 'p', 'y', or a list of these).
            data (float or list): The attitude adjustment amount(s).

        Raises:
            ValueError: If the length of direction and data don't match when using a list.
        """
        if isinstance(direction, list):
            if len(direction) != len(data):
                print("ERROR!The length of direction and data don't match!")
                return
            for i in range(len(data)):
                self.__attitude(direction[i], data[i])
        else:
            self.__attitude(direction, data)

    def action(self, action_id, wait=False):
        """
        Makes the XGO robot perform a predefined action.

        Parameters:
            action_id (int): The ID of the action to perform (1-255).
            wait (bool, optional): If True, the program will wait for the action to complete before proceeding. Defaults to False.

        Raises:
            ValueError: If an invalid action ID is provided.
        """
        if action_id <= 0 or action_id > 255:
            print("ERROR!Illegal Action ID!")
            return
        XGOorder["ACTION"][1] = action_id
        self.__send("ACTION")
        if wait:
            st = XGOparam["ActionTime"].get(action_id)
            if st:
                time.sleep(st)

    def reset(self):
        """
        Resets the XGO robot to its initial state.
        """
        self.action(255)
        time.sleep(1)

    def leg(self, leg_id, data):
        """
        Controls the three-axis movement of a single leg of the XGO robot.

        Parameters:
            leg_id (int): The ID of the leg to control (1, 2, 3, or 4).
            data (list): A list of three float values representing the x, y, and z coordinates of the leg's end effector.

        Raises:
            ValueError: If an invalid leg ID or data length is provided.
        """
        value = [0, 0, 0]
        if leg_id not in [1, 2, 3, 4]:
            print("Error!Illegal Index!")
            return
        if len(data) != 3:
            message = "Error!Illegal Value!"
            return
        for i in range(3):
            try:
                value[i] = conver2u8(data[i], XGOparam["LEG_LIMIT"][i])
            except:
                print("Error!Illegal Value!")
        for i in range(3):
            index = 3 * (leg_id - 1) + i + 1
            XGOorder["LEG_POS"][index] = value[i]
            self.__send("LEG_POS", index)

    def __motor(self, index, data):
        """
        Controls a single motor of the XGO robot.

        Parameters:
            index (int): The index of the motor to control (1-15).
            data (float): The target angle for the motor.
        """
        if index < 13:
            XGOorder["MOTOR_ANGLE"][index] = conver2u8(data, XGOparam["MOTOR_LIMIT"][(index - 1) % 3])
        elif index == 13:
            self.claw(conver2u8(data, XGOparam["MOTOR_LIMIT"][3]))
            return
        else:
            XGOorder["MOTOR_ANGLE"][index] = conver2u8(data, XGOparam["MOTOR_LIMIT"][index - 10])
        self.__send("MOTOR_ANGLE", index)

    def motor(self, motor_id, data):
        """
        Controls one or more motors of the XGO robot.

        Parameters:
            motor_id (int or list): The ID(s) of the motor(s) to control (11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, or 53).
            data (float or list): The target angle(s) for the motor(s).

        Raises:
            ValueError: If an invalid motor ID or data length is provided.
        """
        MOTOR_ID = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53]

        if isinstance(motor_id, list):
            if len(motor_id) != len(data):
                print("Error!Length Mismatching!")
                return
            index = []
            for i in range(len(motor_id)):
                temp_index = search(motor_id[i], MOTOR_ID)
                if temp_index == -1:
                    print("Error!Illegal Index!")
                    return
                index.append(temp_index)
            for i in range(len(index)):
                self.__motor(index[i], data[i])
        else:
            index = search(motor_id, MOTOR_ID)
            self.__motor(index, data)

    def unload_motor(self, leg_id):
        """
        Unloads the motors of a specified leg.

        Parameters:
            leg_id (int): The ID of the leg (1, 2, 3, 4, or 5 for all legs).

        Raises:
            ValueError: If an invalid leg ID is provided.
        """
        if leg_id not in [1, 2, 3, 4, 5]:
            print('ERROR!leg_id must be 1, 2, 3 ,4 or 5')
            return
        XGOorder["UNLOAD_MOTOR"][1] = 0x10 + leg_id
        self.__send("UNLOAD_MOTOR")

    def unload_allmotor(self):
        """
        Unloads all motors of the XGO robot.
        """
        XGOorder["UNLOAD_MOTOR"][1] = 0x01
        self.__send("UNLOAD_MOTOR")

    def load_motor(self, leg_id):
        """
        Loads the motors of a specified leg.

        Parameters:
            leg_id (int): The ID of the leg (1, 2, 3, 4, or 5 for all legs).

        Raises:
            ValueError: If an invalid leg ID is provided.
        """
        if leg_id not in [1, 2, 3, 4, 5]:
            print('ERROR!leg_id must be 1, 2, 3 ,4 or 5')
            return
        XGOorder["LOAD_MOTOR"][1] = 0x20 + leg_id
        self.__send("LOAD_MOTOR")

    def load_allmotor(self):
        """
        Loads all motors of the XGO robot.
        """
        XGOorder["LOAD_MOTOR"][1] = 0x00
        self.__send("LOAD_MOTOR")

    def __periodic_rot(self, direction, period):
        """
        Initiates periodic rotation of the XGO robot's body along a specified axis.

        Parameters:
            direction (str): The axis of rotation ('r' for roll, 'p' for pitch, or 'y' for yaw).
            period (float): The period of rotation.
        """
        index = search(direction, ['r', 'p', 'y'])
        if index == -1:
            print("ERROR!Direction must be 'r', 'p' or 'y'")
            return
        if period == 0:
            XGOorder["PERIODIC_ROT"][index] = 0
        else:
            XGOorder["PERIODIC_ROT"][index] = conver2u8(period, XGOparam["PERIOD_LIMIT"][0], min_value=1)
        self.__send("PERIODIC_ROT", index)

    def periodic_rot(self, direction, period):
        """
        Initiates periodic rotation of the XGO robot's body.

        Parameters:
            direction (str or list): The axis/axes of rotation ('r', 'p', 'y', or a list of these).
            period (float or list): The period(s) of rotation.

        Raises:
            ValueError: If the length of direction and period don't match when using a list.
        """
        if (isinstance(direction, list)):
            if (len(direction) != len(period)):
                print("ERROR!The length of direction and data don't match!")
                return
            for i in range(len(period)):
                self.__periodic_rot(direction[i], period[i])
        else:
            self.__periodic_rot(direction, period)

    def __periodic_tran(self, direction, period):
        """
        Initiates periodic translation of the XGO robot's body along a specified axis.

        Parameters:
            direction (str): The axis of translation ('x', 'y', or 'z').
            period (float): The period of translation.
        """
        index = search(direction, ['x', 'y', 'z'])
        if index == -1:
            print("ERROR!Direction must be 'x', 'y' or 'z'")
            return
        if period == 0:
            XGOorder["PERIODIC_TRAN"][index] = 0
        else:
            XGOorder["PERIODIC_TRAN"][index] = conver2u8(period, XGOparam["PERIOD_LIMIT"][0], min_value=1)
        self.__send("PERIODIC_TRAN", index)

    def periodic_tran(self, direction, period):
        """
        Initiates periodic translation of the XGO robot's body.

        Parameters:
            direction (str or list): The axis/axes of translation ('x', 'y', 'z', or a list of these).
            period (float or list): The period(s) of translation.

        Raises:
            ValueError: If the length of direction and period don't match when using a list.
        """
        if isinstance(direction, list):
            if len(direction) != len(period):
                print("ERROR!The length of direction and data don't match!")
                return
            for i in range(len(period)):
                self.__periodic_tran(direction[i], period[i])
        else:
            self.__periodic_tran(direction, period)

    def mark_time(self, data):
        """
        Makes the XGO robot mark time (原地踏步).

        Parameters:
            data (float): The height of the mark time movement. Set to 0 to stop marking time.
        """
        if data == 0:
            XGOorder["MarkTime"][1] = 0
        else:
            XGOorder["MarkTime"][1] = conver2u8(data, XGOparam["MARK_TIME_LIMIT"], min_value=1)
        self.__send("MarkTime")

    def pace(self, mode):
        """
        Changes the step frequency of the XGO robot.

        Parameters:
            mode (str): The desired pace ('normal', 'slow', or 'high').

        Raises:
            ValueError: If an invalid pace mode is provided.
        """
        if mode == "normal":
            value = 0x00
        elif mode == "slow":
            value = 0x01
        elif mode == "high":
            value = 0x02
        else:
            print("ERROR!Illegal Value!")
            return
        XGOorder["MOVE_MODE"][1] = value
        self.__send("MOVE_MODE")

    def gait_type(self, mode):
        """
        Sets the gait type of the XGO robot.

        Parameters:
            mode (str): The desired gait type ('trot', 'walk', 'high_walk', or 'slow_trot').
        """
        if mode == "trot":
            value = 0x00
        elif mode == "walk":
            value = 0x01
        elif mode == "high_walk":
            value = 0x02
        elif mode == "slow_trot":
            value = 0x03
        XGOorder["GAIT_TYPE"][1] = value
        self.__send("GAIT_TYPE")

    def imu(self, mode):
        """
        Turns on/off the self-stabilization of the XGO robot.

        Parameters:
            mode (int): 1 to turn on self-stabilization, 0 to turn it off.

        Raises:
            ValueError: If an invalid mode value is provided.
        """
        if mode != 0 and mode != 1:
            print("ERROR!Illegal Value!")
            return
        XGOorder["IMU"][1] = mode
        self.__send("IMU")

    def perform(self, mode):
        """
        Turns on/off the XGO robot's performance mode (循环做动作状态).

        Parameters:
            mode (int): 1 to turn on performance mode, 0 to turn it off.

        Raises:
            ValueError: If an invalid mode value is provided.
        """
        if mode != 0 and mode != 1:
            print("ERROR!Illegal Value!")
            return
        XGOorder["PERFORM"][1] = mode
        self.__send("PERFORM")

    def motor_speed(self, speed):
        """
        Adjusts the rotation speed of the motors.

        Parameters:
            speed (int): The desired motor speed (1-255).

        Raises:
            ValueError: If an invalid speed value is provided.
        """
        if speed < 0 or speed > 255:
            print("ERROR!Illegal Value!The speed parameter needs to be between 0 and 255!")
            return
        if speed == 0:
            speed = 1
        XGOorder["MOTOR_SPEED"][1] = speed
        self.__send("MOTOR_SPEED")

    def bt_rename(self, name):
        """
        Renames the Bluetooth name of the XGO robot.

        Parameters:
            name (str): The new Bluetooth name (maximum 10 characters, ASCII only).

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the name is longer than 10 characters or contains non-ASCII characters.
        """
        if type(name) != str:
            print("ERROR!The input value must be of string type!")
            return
        len_name = len(name)
        if len_name > 10:
            print("ERROR!The length of the input string cannot be longer than 10!")
            return
        try:
            XGOorder["BT_NAME"][1:len_name + 1] = list(name.encode('ascii'))
            self.__send("BT_NAME", len=len_name)
        except:
            print("ERROR!Name only supports characters in ASCII code!")

    def read_motor(self):
        """
        Reads the angles of all 15 motors.

        Returns:
            list: A list of 15 float values representing the angles of the motors.
        """
        self.__read(XGOorder["MOTOR_ANGLE"][0], 15)
        angle = []
        if self.__unpack():
            for i in range(self.rx_COUNT + 1):
                if i < 12:
                    angle.append(round(conver2float(self.rx_data[i], XGOparam["MOTOR_LIMIT"][i % 3]), 2))
                else:
                    angle.append(round(conver2float(self.rx_data[i], XGOparam["MOTOR_LIMIT"][i - 9]), 2))
        return angle

    def read_battery(self):
        """
        Reads the battery level of the XGO robot.

        Returns:
            int: The battery level (0-100).
        """
        self.__read(XGOorder["BATTERY"][0], 1)
        battery = 0
        if self.__unpack():
            battery = int(self.rx_data[0])
        return battery

    def read_firmware(self):
        """
        Reads the firmware version of the XGO robot.

        Returns:
            str: The firmware version string.
        """
        self.__read(XGOorder["FIRMWARE_VERSION"][0], 10)
        firmware_version = 'Null'
        if self.__unpack():
            data = self.rx_data[0:10]
            try:
                firmware_version = data.decode("ascii").strip('\0')
            except Exception as e:
                print(e)
        return firmware_version

    def read_roll(self):
        """
        Reads the roll angle of the XGO robot.

        Returns:
            float: The roll angle in degrees.
        """
        self.__read(XGOorder["ROLL"][0], 4)
        roll = 0
        if self.__unpack():
            roll = Byte2Float(self.rx_data)
        return round(roll, 2)

    def read_pitch(self):
        """
        Reads the pitch angle of the XGO robot.

        Returns:
            float: The pitch angle in degrees.
        """
        self.__read(XGOorder["PITCH"][0], 4)
        pitch = 0
        if self.__unpack():
            pitch = Byte2Float(self.rx_data)
        return round(pitch, 2)

    def read_yaw(self):
        """
        Reads the yaw angle of the XGO robot.

        Returns:
            float: The yaw angle in degrees.
        """
        self.__read(XGOorder["YAW"][0], 4)
        yaw = 0
        if self.__unpack():
            yaw = Byte2Float(self.rx_data)
        return round(yaw, 2)

    def __unpack(self, timeout=1):
        """
        Unpacks received data from the XGO robot.

        Parameters:
            timeout (float, optional): The maximum time (in seconds) to wait for a complete data packet. Defaults to 1.

        Returns:
            bool: True if a complete data packet was received and unpacked successfully, False otherwise.
        """
        t = time.time()
        rx_msg = []
        while time.time() - t < timeout:
            n = self.ser.inWaiting()
            rx_CHECK = 0
            if n:
                data = self.ser.read(n)
                for num in data:
                    rx_msg.append(num)
                    if self.rx_FLAG == 0:
                        if num == 0x55:
                            self.rx_FLAG = 1
                        else:
                            self.rx_FLAG = 0

                    elif self.rx_FLAG == 1:
                        if num == 0x00:
                            self.rx_FLAG = 2
                        else:
                            self.rx_FLAG = 0

                    elif self.rx_FLAG == 2:
                        self.rx_LEN = num
                        self.rx_FLAG = 3

                    elif self.rx_FLAG == 3:
                        self.rx_TYPE = num
                        self.rx_FLAG = 4

                    elif self.rx_FLAG == 4:
                        self.rx_ADDR = num
                        self.rx_FLAG = 5
                        self.rx_COUNT = 0

                    elif self.rx_FLAG == 5:
                        if self.rx_COUNT == (self.rx_LEN - 9):
                            self.rx_data[self.rx_COUNT] = num
                            self.rx_FLAG = 6
                        elif self.rx_COUNT < self.rx_LEN - 9:
                            self.rx_data[self.rx_COUNT] = num
                            self.rx_COUNT = self.rx_COUNT + 1

                    elif self.rx_FLAG == 6:
                        for i in self.rx_data[0:(self.rx_LEN - 8)]:
                            rx_CHECK = rx_CHECK + i
                        rx_CHECK = 255 - (self.rx_LEN + self.rx_TYPE + self.rx_ADDR + rx_CHECK) % 256
                        if num == rx_CHECK:
                            self.rx_FLAG = 7
                        else:
                            self.rx_FLAG = 0
                            self.rx_COUNT = 0
                            self.rx_ADDR = 0
                            self.rx_LEN = 0

                    elif self.rx_FLAG == 7:
                        if num == 0x00:
                            self.rx_FLAG = 8
                        else:
                            self.rx_FLAG = 0
                            self.rx_COUNT = 0
                            self.rx_ADDR = 0
                            self.rx_LEN = 0

                    elif self.rx_FLAG == 8:
                        if num == 0xAA:
                            self.rx_FLAG = 0
                            if self.verbose:
                                print("rx_data: ", rx_msg)
                            return True
                        else:
                            self.rx_FLAG = 0
                            self.rx_COUNT = 0
                            self.rx_ADDR = 0
                            self.rx_LEN = 0
        return False

    def set_move_mintime(self, mintime):
        """
        Sets the minimum movement time for the XGO robot.

        Parameters:
            mintime (float): The minimum movement time in seconds.
        """
        self.mintime = mintime

    def upgrade(self, filename):
        """
        Upgrades the firmware of the XGO robot.

        Parameters:
            filename (str): The path to the firmware file.
        """
        XGOorder["UPGRADE"][1] = 1
        self.ser.flush()
        self.__send("UPGRADE")
        if self.__unpack(10):
            if self.rx_data[0] == 0x55:
                time.sleep(1)
                print("Start!")
                self.__send_bin(filename)
            else:
                print("Upgrade Response Error!")
        else:
            print("Upgrade Timeout!")

    def read_lib_version(self):
        """
        Returns the version of the XGO Python library.

        Returns:
            str: The library version string.
        """
        return __version__

    def __send_bin(self, filename):
        """
        Sends a binary file to the XGO robot for firmware upgrade. (TESTING PHASE)

        Parameters:
            filename (str): The path to the binary file.
        """
        try:
            self.__change_baud(350000)
            with open(filename, 'rb') as f:
                file = f.read()
            print("The file size is", len(file), ' bytes.')
            print("The expected upgrade time is", round(len(file) / 350000 * 8 * 1.3), ' s.')
            self.ser.write(file)
            print("Done!")
            self.__change_baud(115200)
        except Exception as e:
            print("Send bin file error!")
            print(e)

    def calibration(self, state):
        """
        Initiates or terminates the calibration process of the XGO robot.

        Parameters:
            state (str): 'start' to initiate calibration, 'end' to terminate.

        Raises:
            ValueError: If an invalid state is provided.
        """
        if state == 'start':
            XGOorder["CALIBRATION"][1] = 1
        elif state == 'end':
            XGOorder["CALIBRATION"][1] = 0
        else:
            print("ERROR!")
        self.__send("CALIBRATION")
        return

    def arm(self, arm_x, arm_z):
        """
        Controls the movement of the XGO robot's arm in the x and z directions.

        Parameters:
            arm_x (float): The x-coordinate of the arm's end effector.
            arm_z (float): The z-coordinate of the arm's end effector.

        Raises:
            ValueError: If invalid arm_x or arm_z values are provided.
        """
        try:
            arm_x_u8 = conver2u8(arm_x, XGOparam["ARM_LIMIT"][0])
            arm_z_u8 = conver2u8(arm_z, XGOparam["ARM_LIMIT"][1])
        except:
            print("Error!Illegal Value!")
            return
        XGOorder["ARM_X"][1] = arm_x_u8
        XGOorder["ARM_Z"][1] = arm_z_u8
        self.__send("ARM_X")
        self.__send("ARM_Z")

    def arm_polar(self, arm_theta, arm_r):
        """
        Controls the movement of the XGO robot's arm using polar coordinates.

        Parameters:
            arm_theta (float): The angle (theta) of the arm.
            arm_r (float): The radial distance (r) of the arm's end effector.

        Raises:
            ValueError: If invalid arm_theta or arm_r values are provided.
        """
        try:
            arm_theta_u8 = conver2u8(arm_theta, XGOparam["ARM_LIMIT"][2])
            arm_r_u8 = conver2u8(arm_r, XGOparam["ARM_LIMIT"][3])
        except:
            print("Error!Illegal Value!")
            return
        XGOorder["ARM_THETA"][1] = arm_theta_u8
        XGOorder["ARM_R"][1] = arm_r_u8
        self.__send("ARM_THETA")
        self.__send("ARM_R")

    def arm_mode(self, mode):
        """
        Sets the mode of the XGO robot's arm.

        Parameters:
            mode (int): The arm mode (0x00 or 0x01).

        Raises:
            ValueError: If an invalid mode value is provided.
        """
        if mode != 0x01 and mode != 0x00:
            print("Error!Illegal Value!")
            return
        XGOorder["ARM_MODE"][1] = mode
        self.__send("ARM_MODE")

    def claw(self, pos):
        """
        Controls the position of the XGO robot's claw.

        Parameters:
            pos (float): The desired claw position (0-255).

        Raises:
            ValueError: If an invalid claw position value is provided.
        """
        try:
            claw_pos = conver2u8(pos, [0, 255])
        except:
            print("Error!Illegal Value!")
            return
        XGOorder["CLAW"][1] = claw_pos
        self.__send("CLAW")

    def btRename(self, name):
        """
        Renames the Bluetooth name of the XGO robot (alternative implementation).

        Parameters:
            name (str): The new Bluetooth name (maximum 20 characters, alphanumeric only).

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the name is longer than 20 characters or contains non-alphanumeric characters.
        """
        length = len(name)
        if not isinstance(name, str):
            print("Wrong type!")
            return

        if length > 20:
            print("The length of the name needs to be less than 20")
            return

        if not name.isalnum():
            print("The name can only contain numbers and letters")
            return

        XGOorder["BT_NAME"] = [0x13]
        for c in list(name):
            if ord(c) > 255:
                print("The name can only contain numbers and letters")
                return
            else:
                XGOorder["BT_NAME"].append(ord(c))
        print(XGOorder["BT_NAME"])
        self.__send("BT_NAME", len=length)

    def moveToMid(self):
        """
        Moves the XGO robot's legs to their middle position.
        """
        self.__send("MOVE_TO_MID")

    def teach(self, mode, pos_id):
        """
        Records or plays back a taught position for the XGO robot's legs.

        Parameters:
            mode (str): 'record' to record a position, 'play' to play back a recorded position.
            pos_id (int): The ID of the position to record or play back.
        """
        if mode == "play":
            XGOorder["TEACH_PLAY"][1] = pos_id
            self.__send("TEACH_PLAY")
        if mode == "record":
            XGOorder["TEACH_RECORD"][1] = pos_id
            self.__send("TEACH_RECORD")
        else:
            return

    def teach_arm(self, mode, pos_id):
        """
        Records or plays back a taught position for the XGO robot's arm.

        Parameters:
            mode (str): 'record' to record a position, 'play' to play back a recorded position.
            pos_id (int): The ID of the position to record or play back.
        """
        if mode == "play":
            XGOorder["TEACH_ARM_PLAY"][1] = pos_id
            self.__send("TEACH_ARM_PLAY")
        if mode == "record":
            XGOorder["TEACH_ARM_RECORD"][1] = pos_id
            self.__send("TEACH_ARM_RECORD")
        else:
            return

    def arm_speed(self, speed):
        """
        Adjusts the rotation speed of the arm.

        Parameters:
            speed (int): The desired arm speed (1-255).

        Raises:
            ValueError: If an invalid speed value is provided.
        """
        if speed < 0 or speed > 255:
            print("ERROR!Illegal Value!The speed parameter needs to be between 0 and 255!")
            return
        if speed == 0:
            speed = 1
        XGOorder["ARM_SPEED"][1] = speed
        self.__send("ARM_SPEED")

    def read_imu(self):
        """
        Reads IMU (Inertial Measurement Unit) data from the XGO robot.

        Returns:
            list: A list containing 9 float values: [acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw].
        """
        self.__read(0x65, 24)
        result = []
        if self.__unpack():
            result = self.unpack_imu()
        return result

    def read_imu_int16(self, direction):
        """
        Reads IMU data as signed 16-bit integers for a specific direction.

        Parameters:
            direction (str): The direction to read ('roll', 'pitch', or 'yaw').

        Returns:
            int or None: The IMU value as a signed 16-bit integer, or None if an invalid direction is provided.
        """
        if direction == "roll":
            self.__read(0x66, 2)
        elif direction == "pitch":
            self.__read(0x67, 2)
        elif direction == "yaw":
            self.__read(0x68, 2)
        else:
            return None
        result = []
        if self.__unpack():
            result = Byte2Short(self.rx_data)
        return result

    def unpack_imu(self):
        """
        Unpacks raw IMU data into meaningful values.

        Returns:
            list: A list containing 9 float values: [acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw].
        """
        result = []
        for i in range(9):
            a = bytearray()
            if i < 6:
                a.append(self.rx_data[2 * i + 1])
                a.append(self.rx_data[2 * i])
                if i < 3:
                    result.append(struct.unpack("!h", a)[0] / 16384 * 9.8)
                else:
                    result.append(struct.unpack("!h", a)[0] / 16.4)
            else:
                a.append(self.rx_data[4 * i - 9])
                a.append(self.rx_data[4 * i - 10])
                a.append(self.rx_data[4 * i - 11])
                a.append(self.rx_data[4 * i - 12])
                result.append(struct.unpack("!f", a)[0] / 180 * 3.14)
        return result

    def set_origin(self):
        """
        Sets the current position of the XGO robot as the origin.
        """
        XGOorder["SET_ORIGIN"][1] = 1
        self.__send("SET_ORIGIN")

    def move_to(self, data):
        """
        Moves the robot to a specified position.

        Parameters:
            data (int): The target position value.
        """
        packed_data = struct.pack('>h', data)
        XGOorder["MOVE_TO"][1] = packed_data[0]
        XGOorder["MOVE_TO"][2] = packed_data[1]
        self.__send("MOVE_TO", len=2)

    def output_analog(self, data):
        """
        Sets the analog output value.

        Parameters:
            data (int): The analog output value.
        """
        XGOorder["OUTPUT_ANALOG"][1] = data
        self.__send("OUTPUT_ANALOG")
        pass

    def output_digital(self, data):
        """
        Sets the digital output value.

        Parameters:
            data (int): The digital output value.
        """
        XGOorder["OUTPUT_DIGITAL"][1] = data
        self.__send("OUTPUT_DIGITAL")
        pass

    ############# RIDER ################

    def rider_move_x(self, speed, runtime=0):
        """
        Moves the XGO Rider along the x-axis.

        Parameters:
            speed (float): The speed of movement along the x-axis.
            runtime (float, optional): The duration of the movement in seconds. If provided, the robot will stop moving after this duration. Defaults to 0.
        """
        XGOorder["VX"][1] = conver2u8(speed, XGOparam["VX_LIMIT"])
        self.__send("VX")
        if runtime:
            time.sleep(runtime)
            XGOorder["VX"][1] = conver2u8(0, XGOparam["VX_LIMIT"])
            self.__send("VX")

    def rider_turn(self, speed, runtime=0):
        """
        Turns the XGO Rider.

        Parameters:
            speed (float): The speed of the turn.
            runtime (float, optional): The duration of the turn in seconds. If provided, the robot will stop turning after this duration. Defaults to 0.
        """
        XGOorder["VYAW"][1] = conver2u8(speed, XGOparam["VYAW_LIMIT"])
        self.__send("VYAW")
        if runtime:
            time.sleep(runtime)
            XGOorder["VYAW"][1] = conver2u8(0, XGOparam["VYAW_LIMIT"])
            self.__send("VYAW")

    def rider_reset_odom(self):
        """
        Resets the odometry of the XGO Rider.
        """
        XGOorder["SET_ORIGIN"][1] = 1
        self.__send("SET_ORIGIN")

    def rider_action(self, action_id, wait=False):
        """
        Makes the XGO Rider perform a predefined action.

        Parameters:
            action_id (int): The ID of the action to perform (1-255).
            wait (bool, optional): If True, the program will wait for the action to complete before proceeding. Defaults to False.

        Raises:
            ValueError: If an invalid action ID is provided.
        """
        if action_id <= 0 or action_id > 255:
            print("ERROR!Illegal Action ID!")
            return
        XGOorder["ACTION"][1] = action_id
        self.__send("ACTION")
        if wait:
            st = XGOparam["ActionTime"].get(action_id)
            if st:
                time.sleep(st)

    def rider_balance_roll(self, mode):
        """
        Turns on/off the roll balance of the XGO Rider.

        Parameters:
            mode (int): 1 to turn on roll balance, 0 to turn it off.

        Raises:
            ValueError: If an invalid mode value is provided.
        """
        if mode != 0 and mode != 1:
            print("ERROR!Illegal Value!")
            return
        XGOorder["IMU"][1] = mode
        self.__send("IMU")

    def rider_perform(self, mode):
        """
        Turns on/off the XGO Rider's performance mode.

        Parameters:
            mode (int): 1 to turn on performance mode, 0 to turn it off.

        Raises:
            ValueError: If an invalid mode value is provided.
        """
        if mode != 0 and mode != 1:
            print("ERROR!Illegal Value!")
            return
        XGOorder["PERFORM"][1] = mode
        self.__send("PERFORM")

    def rider_calibration(self, state):
        """
        Initiates or terminates the calibration process of the XGO Rider.

        Parameters:
            state (str): 'start' to initiate calibration, 'end' to terminate.

        Raises:
            ValueError: If an invalid state is provided.
        """
        if state == 'start':
            XGOorder["CALIBRATION"][1] = 1
        elif state == 'end':
            XGOorder["CALIBRATION"][1] = 0
        else:
            print("ERROR!")
        self.__send("CALIBRATION")
        return

    def rider_height(self, data):
        """
        Adjusts the height of the XGO Rider.

        Parameters:
            data (float): The desired height.
        """
        self.__translation("z", data)

    def rider_roll(self, data):
        """
        Adjusts the roll angle of the XGO Rider.

        Parameters:
            data (float): The desired roll angle.
        """
        self.__attitude("r", data)

    def rider_periodic_roll(self, period):
        """
        Initiates periodic roll movement of the XGO Rider.

        Parameters:
            period (float): The period of the roll movement.
        """
        self.__periodic_rot("r", period)

    def rider_periodic_z(self, period):
        """
        Initiates periodic vertical movement of the XGO Rider.

        Parameters:
            period (float): The period of the vertical movement.
        """
        self.__periodic_tran("z", period)

    def rider_read_battery(self):
        """
        Reads the battery level of the XGO Rider.

        Returns:
            int: The battery level (0-100).
        """
        self.__read(XGOorder["BATTERY"][0], 1)
        battery = 0
        if self.__unpack():
            battery = int(self.rx_data[0])
        return battery

    def rider_read_firmware(self):
        """
        Reads the firmware version of the XGO Rider.

        Returns:
            str: The firmware version string.
        """
        self.__read(XGOorder["FIRMWARE_VERSION"][0], 10)
        firmware_version = 'Null'
        if self.__unpack():
            data = self.rx_data[0:10]
            try:
                firmware_version = data.decode("ascii").strip('\0')
            except Exception as e:
                print(e)
        return firmware_version

    def rider_read_roll(self):
        """
        Reads the roll angle of the XGO Rider.

        Returns:
            float: The roll angle in degrees.
        """
        self.__read(XGOorder["ROLL"][0], 4)
        roll = 0
        if self.__unpack():
            roll = Byte2Float(self.rx_data)
        return round(roll, 2)

    def rider_read_pitch(self):
        """
        Reads the pitch angle of the XGO Rider.

        Returns:
            float: The pitch angle in degrees.
        """
        self.__read(XGOorder["PITCH"][0], 4)
        pitch = 0
        if self.__unpack():
            pitch = Byte2Float(self.rx_data)
        return round(pitch, 2)

    def rider_read_yaw(self):
        """
        Reads the yaw angle of the XGO Rider.

        Returns:
            float: The yaw angle in degrees.
        """
        self.__read(XGOorder["YAW"][0], 4)
        yaw = 0
        if self.__unpack():
            yaw = Byte2Float(self.rx_data)
        return round(yaw, 2)

    def rider_read_imu_int16(self, direction):
        """
        Reads IMU data as signed 16-bit integers for a specific direction on the XGO Rider.

        Parameters:
            direction (str): The direction to read ('roll', 'pitch', or 'yaw').

        Returns:
            int or None: The IMU value as a signed 16-bit integer, or None if an invalid direction is provided.
        """
        if direction == "roll":
            self.__read(0x66, 2)
        elif direction == "pitch":
            self.__read(0x67, 2)
        elif direction == "yaw":
            self.__read(0x68, 2)
        else:
            return None
        result = []
        if self.__unpack():
            result = Byte2Short(self.rx_data)
        return result

    def rider_reset(self):
        """
        Resets the XGO Rider.
        """
        return self.reset()

    def rider_upgrade(self, filename):
        """
        Upgrades the firmware of the XGO Rider.

        Parameters:
            filename (str): The path to the firmware file.
        """
        XGOorder["UPGRADE"][1] = 1
        self.ser.flush()
        self.__send("UPGRADE")
        if self.__unpack(10):
            if self.rx_data[0] == 0x55:
                time.sleep(1)
                print("Start!")
                self.__send_bin(filename)
            else:
                print("Upgrade Response Error!")
        else:
            print("Upgrade Timeout!")

    def rider_led(self, index, color):
        """
        Sets the color of an LED on the XGO Rider.

        Parameters:
            index (int): The index of the LED (likely 1-4 depending on hardware).
            color (list): A list of three integers representing the RGB color values (0-255 each).
        """
        XGOorder["LED_COLOR"][0] = 0x68 + index
        XGOorder["LED_COLOR"][1:4] = color
        self.__send("LED_COLOR", len=3)
