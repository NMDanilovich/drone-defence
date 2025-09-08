import time
from typing import Tuple
import logging

import numpy as np
import zmq

from sources import CarriageController

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

        self.x_pid = PID(kp=0.12, ki=0.01, kd=0.05)
        self.y_pid = PID(kp=0.1, ki=0.01, kd=0.05)

        self.running = False
        self._last_data = None # last data message from ai core 

    def _init_connaction(self, filter_msg:str=""):
        """Setting up connection to proxy
        
        Args:
            filter_msg (str): Filter for ZMQ listner messeges. Can be used for recive message from specific camera.
        """
        self.context = zmq.Context.instance()
        self.subscriber = self.context.socket(zmq.SUB)
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

    def run(self):
        logging.info("Controller initialization...")
        self._init_connaction()
        self.running = True
        
        try:
            while self.running:
                # moving, *position = self.controller.get_move_info()
                new_message, tracked, absolute, bbox, error, det_time = self.get_object_info()
                
                if new_message:
                    print(tracked)
                    if tracked:
                        x_error = error[0]
                        y_error = -1 * error[1]
                    
                        x_output = self.x_pid.update(x_error, det_time)
                        y_output = self.y_pid.update(y_error, det_time)

                        print(f"x: {x_error} -> {x_output}")
                        print(f"y: {y_error} -> {y_output}")

                        self.controller.move_relative(x_output, y_output)
                    else:
                        self.controller.move_to_absolute(*absolute)

                        self.x_pid.reset()
                        self.y_pid.reset()


        except KeyboardInterrupt:
            self.running = False
            logging.exception("Keyboard exit.")
        
        finally:
            self.context.destroy()

if __name__ == "__main__":
    TrackingSystem().run()
