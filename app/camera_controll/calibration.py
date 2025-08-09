import argparse

import cv2

from sources import CarriageController

def interactive_mode():
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
    controller = CarriageController()

    print(f"Position: {controller.get_position()}")
    print()
    controller.move_to_start()
    controller.save_position()

    print(f"New position: {controller.get_position()}")

def move_mode(x_steps:int=0, y_degrees:int=0, absolute:bool=False):
    """Main function for hand testing
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python3 calibration.py")
    parser.add_argument("--x", type=int, default=0, help="flag to x axis control")
    parser.add_argument("--y", type=int, default=0, help="flag to y axis control")
    parser.add_argument("--abs", action="store_true", help="use this when you need absolute movement")
    parser.add_argument("--start", action="store_true", help="move to start position")
    parser.add_argument("--inter", action="store_true", help="for interactive mode relative movement")

    args = parser.parse_args()
    
    if args.inter:
        interactive_mode()
    elif args.start:
        start_mode()
    else:
        move_mode(args.x, args.y, args.abs)