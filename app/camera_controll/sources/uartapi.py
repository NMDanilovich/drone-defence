import time
import argparse
from threading import Thread

import serial

from .logs import get_logger

logger = get_logger("Uart", terminal=False)

JETSON_SERIAL = '/dev/ttyTHS0' # for Jetson UART
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
            timeout=1
        )

        self.is_blocking = is_blocking
        self.__executed = True
        self.__results = []
        
    def __sender(self, command:str, end_marker:str=None):
        """Hiden sender.
        Args:
            command (str): command for sending to uart
            end_marker (str): marker of end recieve message. If None, will be received the one message. Default None.
        """
        self.__results = []
        self.__executed = False

        try:
            self.port.write(command.encode())

            time.sleep(0.002)

            while True:
                response = self.port.readline().decode().strip()

                if response == "":
                    logger.warning("None controller answer.")
                    self.__executed = True
                    break
                else:
                    logger.info("Controller answer: %s", response)
                    self.__results.append(response)

                if end_marker is None or end_marker in response:
                    self.__executed = True
                    break
                    
        except Exception as error:
            logger.error("Sender: %s", error)
            
        return self.__results
    
    def exec_status(self):
        return self.__executed

    def get_info(self):
        """Getting controller information
        """

        command = "STATUS\n"
        end_marker = "FIRING"

        return self.__sender(command, end_marker)

    def zero_x_coordinates(self):
        """Zeroing X coordinates on the controller
        """

        command = "ZERO_X\n"
        
        return self.__sender(command)
        
        
    def send_relative(self, x_angle, y_angle):
        """Send relative coordinates on controller

        Args:
            x_angle (float): Degrees of the yaw axis
            y_angle (float): Degrees of the pithc axis
        """

        command = f"XR{x_angle} YR{y_angle}\n"
        end_marker = "TIME"

        return self.__sender(command, end_marker)
    
    def send_absolute(self, x_angle, y_angle):
        """Send absolute coordinates on controller

        Args:
            x_angle (float): Degrees of the yaw axis
            y_angle (float): Degrees of the pithc axis
        """

        command = f"XA{x_angle} YA{y_angle}\n"
        end_marker = "TIME"
            
        return self.__sender(command, end_marker)

    def fire_control(self, mode):
        if mode == "fire":
            command = "FIRE:ON\n"
        elif mode == "stop":    
            command = "FIRE:OFF\n"

        return self.__sender(command)


def main(x_degrees:float=0, y_degrees:float=0):
    """Main function for hand testing
    """
    
    uart = Uart(is_blocking=True)
    
    # print(uart.get_info())
    uart.send_relative(1200, 0)
    for i in range(20):
        time.sleep(0.2)
        print(uart.get_info())
    print(uart.get_info())
    #uart.zero_x_coordinates()
    #uart.send_absolute(x_degrees, y_degrees)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=0)
    parser.add_argument("--on")
    parser.add_argument("--off")

    args = parser.parse_args()
    
    #main()
    main(args.x, args.y)

