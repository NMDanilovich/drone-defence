import logging
import time

import serial

from utils import threaded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JETSON_SERIAL = '/dev/ttyTHS1' # for Jetson UART
DEFAULT_BAUDRATE = 115200

class Uart:
    """Based Uart connection class"""
    def __init__(self, port:str=JETSON_SERIAL, baudrate:int=DEFAULT_BAUDRATE, is_blocking=True):
        """The connection to the serial port is initialized.

        Args:
            port (str, optional): Path to serial port device. Defaults to JETSON_SERIAL ('/dev/ttyTHS1').
            baudrate (int, optional): Baudrate of connaction. Defaults to DEFAULT_BAUDRATE (115200).
            is_blocking (bool, optional): Command sending mode. Defaults to True.
        """
        # sittings UART
        self.port = serial.Serial(
            port=port,  
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )

        self.is_blocking = is_blocking

    def send_coordinates(self, x_steps, y_angle):
        """Send coordinates on Arduino

        Args:
            x_steps (int): Number of steps of the yaw axis stepper motor
            y_angle (int): Degrees of the pithc axis
        
        Examples:
        >>> uart = Uart()
        >>> x_step = 1000
        >>> y_degrees = 45
        >>> uart.send_coordinates(x_step, y_degrees)
        """

        @threaded(daemon=(not self.is_blocking))
        def sender():
            try:
                command = f"X{x_steps} Y{y_angle}\n"
                print(command)
                self.port.write(command.encode())
                
                # Ждем подтверждения
                response = self.port.readline().decode().strip()
                logger.info("Arduino answer: %s", response)
            except Exception as error:
                logger.error("SenderError: %s", error)

        return sender()


if __name__ == "__main__":
    uart = Uart()
    x_step = 1000
    y_degrees = 45
    uart.send_coordinates(x_step, y_degrees)