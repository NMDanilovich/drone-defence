"""
This module implements the tracking system for the drone defence project.

It includes a PID controller for smooth camera movement and a TrackingSystem
class that orchestrates the tracking process based on information from the
AI core.
"""
import time
import datetime
import argparse
from typing import Tuple
from pathlib import Path
import logging

import numpy as np
import zmq
from matplotlib import pyplot as plt

from sources import CarriageController
from visualisation import Visualization
from configs import SystemConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

directory = Path(__file__).parent
RESULTS = directory / "results"
RESULTS.mkdir(exist_ok=True)

class PID:
    """
    A Proportional-Integral-Derivative (PID) controller.

    This class implements a PID controller which is widely used in control
    systems for its simplicity and effectiveness.
    """
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-180, 180),
                 setpoint: float = 0.0):
        """
        Initializes the PID controller.

        Args:
            kp (float): The proportional gain.
            ki (float): The integral gain.
            kd (float): The derivative gain.
            output_limits (Tuple[float, float], optional): The minimum and maximum
                                                          output values.
                                                          Defaults to (-180, 180).
            setpoint (float, optional): The desired setpoint. Defaults to 0.0.
        """
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.setpoint = setpoint
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
        
    def update(self, measured_value: float, measurement_time: float = None) -> float:
        """
        Updates the PID controller with a new measured value.

        Args:
            measured_value (float): The current measured value.
            measurement_time (float, optional): The timestamp of the measurement.
                                                If None, the current time is used.
                                                Defaults to None.

        Returns:
            float: The calculated output value.
        """

        if measurement_time is not None:
            current_time = measurement_time
        else:
            current_time = time.time()

        dt = current_time - self._prev_time
        if dt <= 0:
            return 0.0
            
        error = self.setpoint + measured_value
        
        p_term = self.kp * error
        
        self._integral += error * dt
        i_term = self.ki * self._integral
        
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        self._prev_error = error
        self._prev_time = current_time
        
        return output
        
    def reset(self):
        """Resets the integral and previous error of the PID controller."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
    
    def set_setpoint(self, setpoint: float):
        """
        Sets a new setpoint for the PID controller.

        Args:
            setpoint (float): The new setpoint.
        """
        self.setpoint = setpoint

class TrackingSystem:
    """
    The main class for the tracking system.

    This class controls the camera carriage, receives tracking information
    from the AI core, and uses a PID controller to smoothly track the target.
    """

    def __init__(self, debug=False):
        """
        Initializes the TrackingSystem.

        Args:
            debug (bool, optional): Whether to run in debug mode. Defaults to False.
        """
        self.controller = CarriageController()

        config = SystemConfig()
        self.x_pid = PID(kp=config.PID["x_kp"], ki=config.PID["x_ki"], kd=config.PID["x_kd"])
        self.y_pid = PID(kp=config.PID["y_kp"], ki=config.PID["y_ki"], kd=config.PID["y_kd"])

        self.running = False
        self._last_data = None # last data message from ai core 
        self.debug = debug
        
        if debug:
            logger.setLevel(logging.DEBUG)

    def _init_connaction(self, filter_msg:str=""):
        """
        Sets up the connection to the AI core's publisher.

        Args:
            filter_msg (str, optional): A filter for ZMQ subscriber messages.
                                        Defaults to "".
        """
        self.context = zmq.Context.instance()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.CONFLATE, 1)
        self.subscriber.connect(f"tcp://127.0.0.1:8000")
        self.subscriber.subscribe(filter_msg)

    def get_object_info(self):
        """
        Receives and parses object information from the AI core.

        Returns:
            tuple: A tuple containing:
                - new_msg (bool): Whether the message is new.
                - tracked (bool): Whether the object is being tracked.
                - absolute (tuple): The absolute coordinates of the object.
                - bbox (list): The bounding box of the object.
                - error (tuple): The tracking error.
                - time (float): The timestamp of the detection.
        """
        data = self.subscriber.recv_json()

        tracked = data["tracked"]
        absolute = data["abs"]
        bbox = data["box"]
        error = data["error"]
        time = data["time"]

        if data == self._last_data:
            new_msg = False
        else:
            new_msg = True

        self._last_data = data

        return new_msg, tracked, absolute, bbox, error, time

    def save_results(self):
        """Saves the PID controller's performance data to a plot."""
        x = np.arange(0, self._num_tracked)
        plt.plot(x, self._x_errors, color="brown", label='x error')
        plt.plot(x, self._x_signals, color="lime", label='x signal')
        plt.plot(x, self._y_errors, color="red", label='y error')
        plt.plot(x, self._y_signals, color="green", label='y signal')
        plt.plot(x, np.zeros_like(x), color="blue", label='target')
        plt.grid()
        plt.legend()
        plt.savefig(RESULTS.joinpath("pid.png"))  

    def run(self):
        """
        The main loop of the tracking system.

        This method continuously receives tracking information and controls
        the camera carriage to follow the target.
        """
        logging.info("Controller initialization...")
        self._init_connaction()
        self.running = True
        plt.plot()

        self._y_errors = []
        self._x_errors = []
        self._y_signals = []
        self._x_signals = []
        self._num_tracked = 0

        try:
            while self.running:
                # moving, *position = self.controller.get_move_info()
                new_message, tracked, absolute, bbox, error, det_time = self.get_object_info()
                
                if new_message:
                    if tracked:
                        x_error = error[0]
                        y_error = -1 * error[1]

                        self._y_errors.append(y_error)
                        self._x_errors.append(x_error)

                        x_output = self.x_pid.update(x_error)
                        y_output = self.y_pid.update(y_error)

                        logger.debug(f"x: {x_error} -> {x_output}")
                        logger.debug(f"y: {y_error} -> {y_output}")
                        
                        dt_object = datetime.datetime.fromtimestamp(det_time)
                        logger.debug(f"Tracked time: {dt_object} Current time: {datetime.datetime.now()}")

                        self._y_signals.append(y_output)
                        self._x_signals.append(x_output)
                        self._num_tracked += 1
                        
                        self.controller.move_relative(x_output, y_output)

                    else:
                        self.controller.move_to_absolute(*absolute)

                        self.x_pid.reset()
                        self.y_pid.reset()
            
                    # if self._num_tracked == 100:
                    #     break 

        except KeyboardInterrupt:
            self.running = False
            logging.exception("Keyboard exit.")
        
        finally:
            self.save_results()
            self.context.destroy()

def start_system(core=False, debug=False):
    """
    Starts the tracking system and optionally the AI core and visualization.

    Args:
        core (bool, optional): Whether to start the AI core. Defaults to False.
        debug (bool, optional): Whether to start the visualization. Defaults to False.
    """

    if debug:
        vis = Visualization()
        vis.start()

    if core:
        from core import AICore
        ai_core = AICore()
        ai_core.start()

    system = TrackingSystem()
    system.run()

    if debug:
        vis.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    start_system(args.core, args.debug)
    