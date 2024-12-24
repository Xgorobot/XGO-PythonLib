import os
import sys
import time
import spidev
import logging
import numpy as np

class RaspberryPi:
    """
    This class provides an interface for controlling hardware peripherals on a Raspberry Pi, specifically SPI, GPIO, and PWM, for use with devices like LCD displays.

    Attributes:
        RST_PIN (int): GPIO pin number for the reset signal.
        DC_PIN (int): GPIO pin number for the data/command signal.
        BL_PIN (int): GPIO pin number for the backlight control signal.
        SPEED (int): SPI communication frequency.
        BL_freq (int): PWM frequency for backlight control.
        GPIO (RPi.GPIO): The RPi.GPIO module for GPIO control.
        SPI (spidev.SpiDev or None): The spidev.SpiDev object for SPI communication, or None if not used.
        _pwm (RPi.GPIO.PWM): The PWM object for backlight control.
    """
    def __init__(self, spi=spidev.SpiDev(0, 0), spi_freq=40000000, rst=27, dc=25, bl=0, bl_freq=1000, i2c=None, i2c_freq=100000):
        """
        Initializes the RaspberryPi object and configures GPIO, SPI, and PWM.

        Parameters:
            spi (spidev.SpiDev, optional): The SPI device object. Defaults to spidev.SpiDev(0, 0). Set to None to disable SPI.
            spi_freq (int, optional): The SPI communication frequency in Hz. Defaults to 40000000.
            rst (int, optional): The GPIO pin number for the reset signal. Defaults to 27.
            dc (int, optional): The GPIO pin number for the data/command signal. Defaults to 25.
            bl (int, optional): The GPIO pin number for the backlight control signal. Defaults to 0.
            bl_freq (int, optional): The PWM frequency for backlight control in Hz. Defaults to 1000.
            i2c (None, optional): Placeholder for I2C configuration (not implemented in this version). Defaults to None.
            i2c_freq (int, optional): Placeholder for I2C frequency (not implemented in this version). Defaults to 100000.
        """
        import RPi.GPIO
        self.np = np
        self.RST_PIN = rst
        self.DC_PIN = dc
        self.BL_PIN = bl
        self.SPEED = spi_freq
        self.BL_freq = bl_freq
        self.GPIO = RPi.GPIO
        # self.GPIO.cleanup()
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setwarnings(False)
        self.GPIO.setup(self.RST_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.DC_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.BL_PIN, self.GPIO.OUT)
        self.GPIO.output(self.BL_PIN, self.GPIO.HIGH)
        # Initialize SPI
        self.SPI = spi
        if self.SPI != None:
            self.SPI.max_speed_hz = spi_freq
            self.SPI.mode = 0b00

    def digital_write(self, pin, value):
        """
        Writes a digital value to a specified GPIO pin.

        Parameters:
            pin (int): The GPIO pin number.
            value (int): The value to write (HIGH or LOW, 1 or 0).
        """
        self.GPIO.output(pin, value)

    def digital_read(self, pin):
        """
        Reads the digital value from a specified GPIO pin.

        Parameters:
            pin (int): The GPIO pin number.

        Returns:
            int: The digital value read from the pin (HIGH or LOW, 1 or 0).
        """
        return self.GPIO.input(pin)

    def delay_ms(self, delaytime):
        """
        Pauses the program execution for a specified number of milliseconds.

        Parameters:
            delaytime (int): The delay time in milliseconds.
        """
        time.sleep(delaytime / 1000.0)

    def spi_writebyte(self, data):
        """
        Writes a list of bytes to the SPI bus.

        Parameters:
            data (list): A list of bytes (integers) to be transmitted.
        """
        if self.SPI != None:
            self.SPI.writebytes(data)

    def bl_DutyCycle(self, duty):
        """
        Sets the duty cycle of the PWM signal for backlight control.

        Parameters:
            duty (int): The duty cycle percentage (0-100).
        """
        self._pwm.ChangeDutyCycle(duty)

    def bl_Frequency(self, freq):
        """
        Sets the frequency of the PWM signal for backlight control.

        Parameters:
            freq (int): The PWM frequency in Hz.
        """
        self._pwm.ChangeFrequency(freq)

    def module_init(self):
        """
        Initializes the module by setting up GPIO pins and starting PWM for backlight control.

        Returns:
            int: 0 if initialization is successful.
        """
        self.GPIO.setup(self.RST_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.DC_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.BL_PIN, self.GPIO.OUT)
        self._pwm = self.GPIO.PWM(self.BL_PIN, self.BL_freq)
        self._pwm.start(100)
        if self.SPI != None:
            self.SPI.max_speed_hz = self.SPEED
            self.SPI.mode = 0b00
        return 0

    def module_exit(self):
        """
        Cleans up the module by closing the SPI connection, stopping PWM, and resetting GPIO pins.
        """
        logging.debug("spi end")
        if self.SPI != None:
            self.SPI.close()

        logging.debug("gpio cleanup...")
        self.GPIO.output(self.RST_PIN, 1)
        self.GPIO.output(self.DC_PIN, 0)
        self._pwm.stop()
        time.sleep(0.001)
        self.GPIO.output(self.BL_PIN, 1)
        # self.GPIO.cleanup()