import argparse
import time
import logging

from config_utils import CarriageConfig
from .uartapi import Uart, JETSON_SERIAL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf = CarriageConfig()

class CarriageController:
    """
    Gun carriage controller that manages positioning using relative movements
    internally and sends absolute coordinates via UART.
    """
    
    def __init__(self, 
                 uart_path=JETSON_SERIAL, 
                 max_x_steps=conf.MAX_X_COORD, 
                 max_y_angle=conf.MAX_Y_COORD, 
                 min_y_angle=conf.MIN_Y_COORD, 
                 start_x_pos=conf.START_X_POSITION,
                 start_y_pos=conf.START_Y_POSITION,
                 last_x_pos=conf.LAST_X_POSITION,
                 last_y_pos=conf.LAST_Y_POSITION,
        ):
        """
        Initialize the carriage controller.
        
        Args:
            uart_path (str): Path to UART device
            max_x_steps (int): Maximum X axis steps (positive direction)
            max_y_angle (int): Maximum Y axis angle in degrees
            min_y_angle (int): Minimum Y axis angle in degrees
            start_x_pos (int): Default X axis position 
            start_y_pos (int): Default Y axis position 
        """
        # Current absolute position
        self.current_x_steps = last_x_pos  # Current absolute X position in steps
        self.current_y_angle = last_y_pos  # Current absolute Y position in degrees
        
        # Movement limits
        self.max_x_steps = max_x_steps
        self.min_x_steps = -max_x_steps
        self.max_y_angle = max_y_angle
        self.min_y_angle = min_y_angle
        
        # UART communication
        self.uart = Uart(uart_path)

        # setup start position
        self.start_x_pos = start_x_pos
        self.start_y_pos = start_y_pos

        logger.info(f"CarriageController initialized - X range: [{self.min_x_steps}, {self.max_x_steps}], "
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
        new_x = self.current_x_steps + delta_x
        new_y = self.current_y_angle + delta_y
        
        # Check limits
        if new_x < self.min_x_steps or new_x > self.max_x_steps:
            logger.warning(f"X movement blocked: {new_x} exceeds limits [{self.min_x_steps}, {self.max_x_steps}]")
            return False
            
        if new_y < self.min_y_angle or new_y > self.max_y_angle:
            logger.warning(f"Y movement blocked: {new_y} exceeds limits [{self.min_y_angle}, {self.max_y_angle}]")
            return False
        
        # Update current position
        self.current_x_steps = new_x
        self.current_y_angle = new_y
        
        # Send absolute coordinates via UART
        self._send_absolute_position()
        
        logger.debug(f"Relative move: delta_x={delta_x}, delta_y={delta_y} -> "
                    f"absolute: x={self.current_x_steps}, y={self.current_y_angle}")
        
        return True
    
    def move_to_absolute(self, x_steps, y_angle):
        """
        Move carriage to absolute position.
        
        Args:
            x_steps (int): Absolute X position in steps
            y_angle (float): Absolute Y position in degrees
            
        Returns:
            bool: True if movement was successful, False if limits exceeded
        """
        # Check limits
        if x_steps < self.min_x_steps or x_steps > self.max_x_steps:
            logger.warning(f"Absolute X position {x_steps} exceeds limits [{self.min_x_steps}, {self.max_x_steps}]")
            return False
            
        if y_angle < self.min_y_angle or y_angle > self.max_y_angle:
            logger.warning(f"Absolute Y position {y_angle} exceeds limits [{self.min_y_angle}, {self.max_y_angle}]")
            return False
        
        # Update current position
        self.current_x_steps = x_steps
        self.current_y_angle = y_angle
        
        # Send absolute coordinates via UART
        self._send_absolute_position()
        
        logger.debug(f"Absolute move to: x={self.current_x_steps}, y={self.current_y_angle}")
        
        return True
         
    def get_position(self):
        """
        Get current absolute position.
        
        Returns:
            tuple: (x_steps, y_angle)
        """
        return (self.current_x_steps, self.current_y_angle)
              
    def _send_absolute_position(self):
        """Send current absolute position via UART."""
        try:
            self.uart.send_coordinates(self.current_x_steps, self.current_y_angle)
            logger.debug(f"Sent absolute position: X{self.current_x_steps} Y{self.current_y_angle}")
        except Exception as e:
            logger.error(f"Failed to send coordinates via UART: {e}")

    def reset_position(self):
        """Reset position tracking to (0, 0) without moving."""
        self.current_x_steps = self.start_x_pos
        self.current_y_angle = self.start_y_pos
        self.move_to_absolute(self.start_x_pos, self.start_y_pos)
        logger.info("Position reset to start")
    
    def save_position(self, config: CarriageConfig = conf):
        """Write the current values of position in config file.:"""
        
        if isinstance(config, CarriageConfig):
            config.LAST_X_POSITION = self.current_x_steps
            config.LAST_Y_POSITION = self.current_y_angle
            config.write()
        else:
            logger.error(ValueError, "Config object must be the CarriageConfig class")


def main(x_steps:int=0, y_degrees:int=0):
    """Main function for hand testing
    """
    
    # Initialize controller
    controller = CarriageController(
        uart_path=JETSON_SERIAL,
    )

    # Show initial status
    print(f"Position: {controller.get_position()}")
    print()

    # Uncomment for moving to start position
    # controller.move_to_start()

    controller.move_relative(x_steps, y_degrees)
    controller.save_position()

    print(f"New position: {controller.get_position()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--abs", action="store_true")

    args = parser.parse_args()
    
    main(args.x, args.y)