"""This script provides a command-line interface for calibrating and controlling a camera carriage.

It allows for interactive control, moving to a predefined start position, and making absolute or relative movements.
"""
import argparse

from sources import CarriageController

def interactive_mode():
    """Starts interactive mode for manual camera carriage calibration.

    Allows the user to move the carriage up, down, left, or right using keyboard inputs (W, S, A, D).
    The position is saved when the user quits (Q).
    """
    # Initialize controller
    controller = CarriageController()

    # Show initial status
    print(f"Position: {controller.get_position()}")
    print()

    while True:
        x_steps = 0
        y_degrees = 0
        speed = 50
        com = input("A/D/W/S: ").lower()
        if com == "a":
            x_steps += speed
        elif com == "d":
            x_steps -= speed
        elif com == "w":
            y_degrees -= int(speed * 0.1)
        elif com == "s":
            y_degrees += int(speed * 0.1)
        elif com == "q":
            break

        controller.move_relative(x_steps, y_degrees)

    controller.save_position()

def start_mode():
    """Moves the camera carriage to the predefined start position.

    The final position is saved.
    """
    controller = CarriageController()

    print(f"Position: {controller.get_position()}")
    print()
    controller.move_to_start()
    controller.save_position()

    print(f"New position: {controller.get_position()}")

def move_mode(x_steps:int=0, y_degrees:int=0, absolute:bool=False):
    """Moves the camera carriage by a specified amount.

    Can perform either absolute or relative movements based on the provided arguments.

    Args:
        x_steps (int, optional): The number of steps to move on the x-axis. Defaults to 0.
        y_degrees (int, optional): The number of degrees to move on the y-axis. Defaults to 0.
        absolute (bool, optional): If True, moves to the absolute position (x_steps, y_degrees).
                                   If False, performs a relative move. Defaults to False.
    """
    
    # Initialize controller
    controller = CarriageController()

    # Show initial status
    print(f"Position: {controller.get_position()}")
    print()

    if absolute:
        controller.move_to_absolute(x_steps, y_degrees)
    else:
        controller.move_relative(x_steps, y_degrees)
    
    controller.save_position()

    print(f"New position: {controller.get_position()}")

def zero_command():
    # Initialize controller
    controller = CarriageController()

    controller.uart.zero_x_coordinates()

def auto_mode():
    # Initialize controller
    controller = CarriageController()
    
    # Show initial status
    print(f"Moving to start position...")
    print()
    controller.move_to_start()
    
    # TODO make auto calibration

    controller.save_position()

    print(f"New position: {controller.get_position()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python3 calibration.py")
    parser.add_argument("--x", type=int, default=0, help="flag to x axis control")
    parser.add_argument("--y", type=int, default=0, help="flag to y axis control")
    parser.add_argument("--abs", action="store_true", help="use this when you need absolute movement")
    parser.add_argument("--start", action="store_true", help="move to start position")
    parser.add_argument("--inter", action="store_true", help="for interactive mode relative movement")
    parser.add_argument("--zero", action="store_true", help="zeroing x axis coordinate on controller")

    args = parser.parse_args()
    
    if args.inter:
        interactive_mode()
    elif args.start:
        start_mode()
    elif args.zero:
        zero_command()
    else:
        move_mode(args.x, args.y, args.abs)
