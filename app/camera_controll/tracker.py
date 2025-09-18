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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

directory = Path(__file__).parent
results = directory / "results"
results.mkdir(exist_ok=True)

class PID:
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-180, 180),
                 setpoint: float = 0.0):
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.setpoint = setpoint
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
        
    def update(self, measured_value: float, measurement_time: float = None) -> float:

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
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()
    
    def set_setpoint(self, setpoint: float):
        self.setpoint = setpoint

class TrackingSystem:

    def __init__(self):
        self.controller = CarriageController()

        self.x_pid = PID(kp=0.0935, ki=0.0, kd=0.0002)
        self.y_pid = PID(kp=0.1, ki=0.003, kd=0.)

        self.running = False
        self._last_data = None # last data message from ai core 

    def _init_connaction(self, filter_msg:str=""):
        """Setting up connection to proxy
        
        Args:
            filter_msg (str): Filter for ZMQ listner messeges. Can be used for recive message from specific camera.
        """
        self.context = zmq.Context.instance()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.CONFLATE, 1)
        self.subscriber.connect(f"tcp://127.0.0.1:8000")
        self.subscriber.subscribe(filter_msg)

    def get_object_info(self):
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
            x = np.arange(0, self._num_tracked)
            plt.plot(x, self._x_errors, color="brown", label='x error')
            plt.plot(x, self._x_signals, color="lime", label='x signal')
            plt.plot(x, self._y_errors, color="red", label='y error')
            plt.plot(x, self._y_signals, color="green", label='y signal')
            plt.plot(x, np.zeros_like(x), color="blue", label='target')
            plt.legend()
            plt.savefig(results.joinpath("pid.png"))  

    def main(self):
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
            
                    if self._num_tracked == 100:
                        break 

        except KeyboardInterrupt:
            self.running = False
            logging.exception("Keyboard exit.")
        
        finally:
            self.save_results()
            self.context.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    
    vis = Visualization()
    vis.start()

    if args.start:
        from core import AICore
        ai_core = AICore()
        ai_core.start()

    TrackingSystem().main()
    vis.stop()

