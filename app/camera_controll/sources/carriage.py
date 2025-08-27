import argparse
import time
import logging

# -----------------DEBUG---------
# import os
# import sys

# CONFIGS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(str(CONFIGS))
# ---------------------------------

from configs import CarriageConfig
from .uartapi import Uart, JETSON_SERIAL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarriageController:
    """
    Gun carriage controller that manages positioning using relative movements
    internally and sends absolute coordinates via UART.
    """
    
    def __init__(self, is_blocking=True):
        """
        Initialize the carriage controller.
        
        Attributes:
            uart (str): Path to UART device
            max_x_steps (int): Maximum X axis steps (positive direction)
            max_y_angle (int): Maximum Y axis angle in degrees
            min_y_angle (int): Minimum Y axis angle in degrees
            start_x_pos (int): Default X axis position 
            start_y_pos (int): Default Y axis position 
        """
        self.config = CarriageConfig()

        # Current absolute position
        self.current_x_angle = self.config.LAST_X_POSITION # Current absolute X position in steps
        self.current_y_angle = self.config.LAST_Y_POSITION  # Current absolute Y position in degrees

        # Movement limits
        self.max_x_angle = self.config.MAX_X_COORD
        self.min_x_angle = self.config.MIN_X_COORD
        self.max_y_angle = self.config.MAX_Y_COORD
        self.min_y_angle = self.config.MIN_Y_COORD
        
        # UART communication
        self.uart = Uart(self.config.SERIAL_PORT, baudrate=self.config.BAUDRATE, is_blocking=is_blocking)
        
        # Information (status) from controller
        self.contr_info: dict
        self.update_info()

        # setup start position
        self.start_x_pos = self.config.START_X_POSITION
        self.start_y_pos = self.config.START_Y_POSITION
        
        self.command_executed = False

        logger.info(f"CarriageController initialized - X range: [{self.min_x_angle}, {self.max_x_angle}], "
                   f"Y range: [{self.min_y_angle}, {self.max_y_angle}]")

    def move_to_start(self):
        """Moves the carriage platform to the starting position"""
        self.move_to_absolute(self.start_x_pos, self.start_y_pos)

    def move_relative(self, delta_x=0, delta_y=0):
        """
        Move carriage by relative amounts and send absolute position via UART.
        
        Args:
            delta_x (int): Relative movement in X axis (steps)
            delta_y (float): Relative movement in Y axis (degrees)
            
        Returns:
            bool: True if movement was successful, False if limits exceeded
        """
        # Calculate new absolute positions
        new_x = self.current_x_angle + delta_x
        new_y = self.current_y_angle + delta_y
        
        # Check limits
        if self.check_limits(new_x, new_y):
        
            # Update current position
            self.current_x_angle = new_x
            self.current_y_angle = new_y
            
            # Send absolute coordinates via UART
            self.uart.send_relative(delta_x, delta_y)
            self.command_executed = self.uart.exec_status()
            
            logger.debug(f"Relative move: delta_x={delta_x}, delta_y={delta_y} -> "
                        f"absolute: x={self.current_x_angle}, y={self.current_y_angle}")
            
            return True
        else:
            return False
    
    def move_to_absolute(self, x_angle, y_angle):
        """
        Move carriage to absolute position.
        
        Args:
            x_steps (int): Absolute X position in steps
            y_angle (float): Absolute Y position in degrees
            
        Returns:
            bool: True if movement was successful, False if limits exceeded
        """
        # Check limits
        if self.check_limits(x_angle, y_angle):

            # Update current position
            self.current_x_angle = x_angle
            self.current_y_angle = y_angle
            
            # Send absolute coordinates via UART
            self.uart.send_absolute(self.current_x_angle, self.current_y_angle)
            self.command_executed = self.uart.exec_status()
            
            logger.debug(f"Absolute move to: x={self.current_x_angle}, y={self.current_y_angle}")
            
            return True
        else: 
            return False
    
    def update_info(self) -> dict:
        """Recive the status information from controller and return dict (update the self.contr_info variable). 
        """
        temp_info = self.uart.get_info()
        temp_info = temp_info[-1]

        if temp_info.startswith("STATUS"):
            temp_info = temp_info.replace("STATUS", "").split()
            temp_dict = {}
            for msg in temp_info:
                key, value = msg.split(":")
                temp_dict[key] = value

            self.contr_info = temp_dict
        else:
            self.contr_info = {}

        return self.contr_info

    def get_position(self):
        """
        Get current absolute position.
        
        Returns:
            tuple: (x_steps, y_angle)
        """

        self.update_info()

        if self.contr_info:
            self.current_x_angle = float(self.contr_info["X"])
            self.current_y_angle = float(self.contr_info["Y"])

        return (self.current_x_angle, self.current_y_angle)
    
    def fire(self, mode):
        self.uart.fire_control(mode)
              
    def check_limits(self, new_x, new_y):

        if self.min_x_angle != "None" or self.max_x_angle != "None":
            if new_x < self.min_x_angle or new_x > self.max_x_angle:
                logger.warning(f"X movement blocked: {new_x} exceeds limits [{self.min_x_angle}, {self.max_x_angle}]")
                return False
            
        if self.min_y_angle != "None" or self.max_y_angle != "None":  
            if new_y < self.min_y_angle or new_y > self.max_y_angle:
                logger.warning(f"Y movement blocked: {new_y} exceeds limits [{self.min_y_angle}, {self.max_y_angle}]")
                return False
        
        return True

    def save_position(self):
        """Write the current values of position in config file.:"""
        
        self.config.LAST_X_POSITION = self.current_x_angle
        self.config.LAST_Y_POSITION = self.current_y_angle
        self.config.write()

def main(x_steps:int=0, y_degrees:int=0, start:bool=False):
    """Main function for hand testing
    """
    
    # Initialize controller
    controller = CarriageController(True)

    # Show initial status
    # print(f"Position: {controller.get_position()}")
    print()

    if start:
        controller.move_to_start()
    else:
        controller.move_relative(x_steps, y_degrees)
    

    controller.save_position()

    print(f"New position: {controller.get_position()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=0)
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--start", action="store_true")

    args = parser.parse_args()
    
    main(args.x, args.y, args.start)
