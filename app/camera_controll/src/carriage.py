import logging

import torch.nn.functional as F

import cfg.carriage_cfg as carr
from uartapi import Uart, JETSON_SERIAL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarriageController:
    """
    Gun carriage controller that manages positioning using relative movements
    internally and sends absolute coordinates via UART.
    """
    
    def __init__(self, 
                 uart_path=JETSON_SERIAL, 
                 max_x_steps=carr.MAX_X_COORD, 
                 max_y_angle=carr.MAX_Y_COORD, 
                 min_y_angle=carr.MIN_Y_COORD, 
                 step_size=10, 
                 angle_step=1
        ):
        """
        Initialize the carriage controller.
        
        Args:
            uart_path (str): Path to UART device
            max_x_steps (int): Maximum X axis steps (positive direction)
            max_y_angle (int): Maximum Y axis angle in degrees
            min_y_angle (int): Minimum Y axis angle in degrees
            step_size (int): Default step size for X axis movements
            angle_step (float): Default angle step for Y axis movements
        """
        # Current absolute position
        self.current_x_steps = 0  # Current absolute X position in steps
        self.current_y_angle = 0  # Current absolute Y position in degrees
        
        # Movement limits
        self.max_x_steps = max_x_steps
        self.min_x_steps = -max_x_steps
        self.max_y_angle = max_y_angle
        self.min_y_angle = min_y_angle
        
        # Movement parameters
        self.step_size = step_size
        self.angle_step = angle_step
        
        # UART communication
        self.uart = Uart(uart_path)
        
        logger.info(f"CarriageController initialized - X range: [{self.min_x_steps}, {self.max_x_steps}], "
                   f"Y range: [{self.min_y_angle}, {self.max_y_angle}]")
    
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
           
    def set_movement_parameters(self, step_size=None, angle_step=None):
        """
        Update movement parameters.
        
        Args:
            step_size (int): New step size for X axis
            angle_step (float): New angle step for Y axis
        """
        if step_size is not None:
            self.step_size = step_size
            logger.info(f"Step size updated to: {self.step_size}")
            
        if angle_step is not None:
            self.angle_step = angle_step
            logger.info(f"Angle step updated to: {self.angle_step}")
    
    def _send_absolute_position(self):
        """Send current absolute position via UART."""
        try:
            self.uart.send_coordinates(self.current_x_steps, self.current_y_angle)
            logger.debug(f"Sent absolute position: X{self.current_x_steps} Y{self.current_y_angle}")
        except Exception as e:
            logger.error(f"Failed to send coordinates via UART: {e}")

    def reset_position(self):
        """Reset position tracking to (0, 0) without moving."""
        self.current_x_steps = 0
        self.current_y_angle = 0
        logger.info("Position reset to (0, 0)")
    
    def get_status(self):
        """
        Get current status information.
        
        Returns:
            dict: Status information including position, limits, and parameters
        """
        return {
            'position': {
                'x_steps': self.current_x_steps,
                'y_angle': self.current_y_angle
            },
            'limits': {
                'x_range': [self.min_x_steps, self.max_x_steps],
                'y_range': [self.min_y_angle, self.max_y_angle]
            },
            'parameters': {
                'step_size': self.step_size,
                'angle_step': self.angle_step
            },
            'normalized_position': self.get_relative_position_normalized(),
        }

def demo_carriage_controller():
    """Demo function showing CarriageController usage."""
    print("CarriageController Demo")
    print("=" * 50)
    
    # Initialize controller
    controller = CarriageController(
        uart_path=JETSON_SERIAL,
        max_x_steps=5000,
        max_y_angle=45,
        min_y_angle=-45,
        step_size=50,
        angle_step=2
    )
    
    # Show initial status
    print("Initial status:")
    status = controller.get_status()
    print(f"Position: {status['position']}")
    print(f"Limits: {status['limits']}")
    print()
    
    # Example movements
    print("Moving to absolute position (500, 10)...")
    controller.move_to_absolute(500, 10)
    print(f"New position: {controller.get_position()}")


    print("\nDemo completed!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_carriage_controller()