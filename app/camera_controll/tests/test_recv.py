import time

import numpy as np
import zmq

from sources.tracked_obj import TrackObject

def send_msg_for_testing():

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 10)
    socket.setsockopt(zmq.RCVHWM, 10)
    socket.bind(f"tcp://127.0.0.1:8000")

    # abs_x = np.random.randint(0, 360)
    # abs_y = np.random.randint(90, 140)

    abs_x = 140
    abs_y = 119

    bbox = [400, 500, 200, 100]
    
    while True:
        absolute = np.array([abs_x, abs_y]) + np.random.randn(2)
        time.sleep(0.016)
        test_target = TrackObject(tuple(absolute), bbox)
        print(test_target)

        message = test_target.__dict__
        socket.send_json(message)

        time.sleep(1)



if __name__ == "__main__":
    send_msg_for_testing()
    