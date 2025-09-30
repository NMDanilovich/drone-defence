import time
import logging
from pathlib import Path

from multiprocessing import Process

import zmq

class TestReceive(Process):
    def __init__(self):
        super().__init__()
        
        self.ctx: zmq.Context = None 
        self.subscriber: zmq.Socket = None
        
        self.msgs = dict()
        self.running = True
    
    def setting(self, filter_msg:str=""):
        """Setting up connection to proxy
        
        Args:
            filter_msg (str): Filter for ZMQ listner messeges. Can be used for recive message from specific camera.
        """
        self.ctx = zmq.Context.instance()
        self.subscriber = self.ctx.socket(zmq.SUB)
        # self.subscriber.setsockopt(zmq.CONFLATE, 1)
        self.subscriber.connect(f"tcp://127.0.0.1:8000")
        # self.subscriber.subscribe("overview")
        self.subscriber.subscribe("tracking")
      
    def run(self):
        self.setting()
        last_data = None
        try:
            while self.running:
                data = self.subscriber.recv_multipart()
                # if data != last_data:
                #     print(data["error"])
                #     print(data["time"])
                #     print(time.time() - data["time"])
                print(data[0], data[1])
                last_data = data
                    
        except Exception as error:
            print(error)

        except KeyboardInterrupt:
            print("exit!!!")

        finally:
            self.ctx.destroy()

    def stop(self):
        """Stoped process"""
        self.running = False

if __name__ == "__main__":
    test = TestReceive()
    try:
        test.start()
        test.join()
    except KeyboardInterrupt:
        test.stop()
        test.join()

def test_send():
    pass