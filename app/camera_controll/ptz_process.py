from multiprocessing import Process
import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils import BBox
from hickapi import HickClient
import config
from stream import VideoStream


class PTZTracker(Process):
    def __init__(self, ip, login, password, model_path):
        
        # --- stream template ---
        self.stream_path = f"rtsp://{login}:{password}@{ip}:554/Streaming/channels/101"
        # self.stream_path = "/home/samurai/workspace/examples/traffic2.mp4"

        # --- settings ---
        self.__model = YOLO(model_path, task="detect")
        self.client = HickClient(ip, login, password)

        # --- get camera settings ---
        cap = cv2.VideoCapture(self.stream_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        else:
            raise FileNotFoundError("Stream not found")
        cap.release()

        # --- set image areas ---
        stop_scale = 0.2

        self.stop_box = BBox(self.width // 2, self.height // 2, self.width * stop_scale, self.height * stop_scale)
        self.fast_box = BBox(self.width // 2, self.height // 2, self.width, self.height)

    @staticmethod
    def preprocess(image, flip=0, imgsz:int=512):
            image = cv2.flip(image, flip)
            new_size = (imgsz, imgsz)
            image = cv2.resize(image, new_size)
            return image

    def run(self):
        stream = VideoStream(self.stream_path)
        
        try: 
            while True:
                frame = stream.read()

                if frame is not None:
                    
                    frame = self.preprocess(frame)
                    frame = cv2.rectangle(frame, self.stop_box.xyxy[:2], self.stop_box.xyxy[2:], (0, 0, 245), 5)

                    results = self.__model.predict(frame, device="cuda:0")

                    for res in results[0]:
                        # if select_drone_discriptor == result_object_detection:
                        if res.boxes.cls == 1: #and res.boxes.id == self.select_id:
                            object_class = int(res.boxes.cls)
                            object_bbox = BBox(*res.boxes.xywh[0])
                            
                            frame = cv2.circle(frame, object_bbox[:2], 7, (0, 0, 0), -1)
                            x_move = int((self.stop_box[0] - object_bbox[0]) * -2 * 100 // self.width)
                            y_move = int((self.stop_box[1] - object_bbox[1]) * -2 * 100 // self.height)
                            z_move = int(self.stop_box[3] * 100 // object_bbox[3])

                            if object_bbox in self.stop_box:
                                speed = 0

                            elif object_bbox in self.fast_box:
                                speed = 1

                            x_move *= speed
                            y_move *= speed
                            z_move *= speed
                            break

                    else:
                        x_move = 0
                        y_move = 0
                        z_move = 0

                    # --- move ---
                    print(x_move, y_move, z_move,)
                    self.client.continuous_move(int(x_move), int(y_move), int(z_move))

                    # cv2.imshow("stream", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            stream.stop()
            cv2.destroyAllWindows()

if __name__ =="__main__":
    ip = config.PTZ_IP
    login = config.LOGIN
    password = config.PASSWORD
    model_path = config.MODEL_PATH
    select_id = 1

    tracker = PTZTracker(ip, login, password, model_path)
    tracker.run()