import logging
import time
import argparse

import serial

from threads_utils import threaded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.__executed = False
        
        @threaded(is_blocking=is_blocking)
        def __sender(command:str, end_marker:str=None):
            """Hiden sender.
            Args:
                command (str): command for sending to uart
                end_marker (str): marker of end recieve message. If None, will be received the one message. Default None.
            """
            results = []
            self.__executed = False

            try:
                self.port.write(command.encode())
                while True:
                    response = self.port.readline().decode().strip()

                    if response == "":
                        logger.warning("None controller answer.")
                        break
                    else:
                        logger.info("Controller answer: %s", response)
                        results.append(response)

                    if end_marker is None or end_marker in response:
                        self.__executed = True
                        break
            except Exception as error:
                logger.error("Sender: %s", error)
                
            return results
        
        self.sender = __sender

    def exec_status(self):
        return self.__executed

    def get_controller_coordinates(self):
        """Getting controller information
        """

        command = "STATUS\n"
        end_marker = "FIRE"

        return self.sender(command, end_marker)

    def zero_x_coordinates(self):
        # TODO don't work
        command = "ZERO_X\n"
        
        return self.sender(command)
        
        
    def send_relative(self, x_angle, y_angle):
        """Send relative coordinates on controller

        Args:
            x_angle (float): Degrees of the yaw axis
            y_angle (float): Degrees of the pithc axis
        
        Examples:
        >>> uart = Uart()
        >>> x_degrees = 1000
        >>> y_degrees = 45
        >>> uart.send_coordinates(x_degrees, y_degrees)
        """

        command = f"XR{x_angle} YR{y_angle}\n"
        end_marker = "TIME"
            
        return self.sender(command, end_marker)
    
    def send_absolute(self, x_angle, y_angle):
        """Send absolute coordinates on controller

        Args:
            x_angle (float): Degrees of the yaw axis
            y_angle (float): Degrees of the pithc axis
        
        Examples:
        >>> uart = Uart()
        >>> x_degrees = 1000
        >>> y_degrees = 45
        >>> uart.send_coordinates(x_degrees, y_degrees)
        """

        command = f"XA{x_angle} YA{y_angle}\n"
        end_marker = "TIME"
            
        return self.sender(command, end_marker)

    def fire_control(self, mode):
        if mode == "fire":
            command = "FIRE ON\n"
        elif mode == "stop":    
            command = "FIRE OFF\n"

        return self.sender(command)
        

def main(x_steps:float=0, y_degrees:float=0):
    """Main function for hand testing
    """
    
    uart = Uart(is_blocking=True)
    
    #uart.fire_control("fire")
    #time.sleep(1)
    #uart.fire_control("stop")
    time.sleep(0.01)
    print(uart.exec_status())
    print(uart.get_controller_coordinates())
    print(uart.exec_status())
    time.sleep(0.01)
    # uart.zero_x_coordinates()
    uart.get_controller_coordinates()
    # for i in range(60):
    #     uart.send_relative(x_steps, y_degrees)
    #     time.sleep(0.03)
    
    time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=0)
    parser.add_argument("--on")
    parser.add_argument("--off")

    args = parser.parse_args()
    
    #main()
    main(args.x, args.y)

