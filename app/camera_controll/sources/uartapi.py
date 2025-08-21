import logging
import time
import argparse

import serial

from .threads_utils import threaded

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

    def send_coordinates(self, x_angle, y_angle):
        """Send coordinates on Arduino

        Args:
            x_angle (float): Degrees of the yaw axis
            y_angle (float): Degrees of the pithc axis
        
        Examples:
        >>> uart = Uart()
        >>> x_degrees = 1000
        >>> y_degrees = 45
        >>> uart.send_coordinates(x_degrees, y_degrees)
        """

        @threaded(is_blocking=self.is_blocking)
        def sender():
            try:
                command = f"X {x_angle} Y {y_angle}\n"
                print(command)
                self.port.write(command.encode())
                
                end_marker = "TIME"

                # wait response
                while True:
                    #if self.port.in_waiting > 0:
                    response = self.port.readline().decode().strip()
                    if response != "":
                        logger.info("Uart answer: %s", response)
                    if end_marker in response:
                        break

            except Exception as error:
                logger.error("SenderError: %s", error)

        return sender()
        
    def fire_control(self, mode):
    
        @threaded(is_blocking=self.is_blocking)
        def sender():
            try:
                if mode == "fire":
                    command = "POWER ON\n"
                elif mode == "stop":    
                    command = "POWER OFF\n"
                print(command)
                self.port.write(command.encode())
                
                # Ждем подтверждения
                response = self.port.readline().decode().strip()
                logger.info("Arduino answer: %s", response)
                # time.sleep(0.1)
            except Exception as error:
                logger.error("SenderError: %s", error)

        return sender()
        

def main(x_steps:float=0, y_degrees:float=0):
    """Main function for hand testing
    """
    
    uart = Uart(is_blocking=False)
    
    #uart.fire_control("fire")
    #time.sleep(2)
    #uart.fire_control("stop")

    for i in range(20):
        uart.send_coordinates(x_steps, y_degrees)
        time.sleep(0.02)
    time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--on")
    parser.add_argument("--off")

    args = parser.parse_args()
    
    #main()
    main(args.x, args.y)

