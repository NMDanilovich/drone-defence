from multiprocessing import Process
import time

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F

from utils import BBox, descriptor
from hickapi import HickClient
import config
from stream import VideoStream

DEBUG_DISCRIPTOR = descriptor(cv2.imread("./test_drone.png"))

class PTZTracker(Process):
    def __init__(self, ip, login, password, model_path):
        
        # --- stream template ---
        self.stream_path = f"rtsp://{login}:{password}@{ip}:554/Streaming/channels/101"
        # self.stream_path = "/home/samurai/workspace/examples/traffic2.mp4"

        # --- settings ---
        self.model = YOLO(model_path, task="detect")
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

        # --- PTZ moves ---
        self.move_x = 0
        self.move_y = 0
        self.move_z = 0

        # --- drone information ---
        self.tracked = False
        self.descriptor = None
        self.actual_id = None
        self.threshold = 0.7


    @staticmethod
    def preprocess(image, flip=0, imgsz:int=512):
            image = cv2.flip(image, flip)
            new_size = (imgsz, imgsz)
            image = cv2.resize(image, new_size)
            return image


    def get_info(self):
        #TODO write the code for recieve information from overview module
        return True, DEBUG_DISCRIPTOR, 1

    def update_id(self, frame, results):
        """Update actual object id"""
        x1, y1, x2, y2 = results.boxes.xyxy
        object_descriptor = descriptor(frame[x1:x2, y1:y2])
    
        distance = F.cosine_similarity(self.descriptor, object_descriptor, dim=0)

        if distance > self.threshold:
            self.actual_id = results.boxes.id

    def update_moves(self, results):
        object_bbox = BBox(*results.boxes.xywh[0])
        if self.actual_id == results.boxes.id:
            # frame = cv2.circle(frame, object_bbox[:2], 7, (0, 0, 0), -1)
            self.x_move = int((self.stop_box[0] - object_bbox[0]) * -2 * 100 // self.width)
            self.y_move = int((self.stop_box[1] - object_bbox[1]) * -2 * 100 // self.height)
            self.z_move = int(self.stop_box[3] * 100 // object_bbox[3])

        if object_bbox in self.stop_box:
            speed = 0.1

        elif object_bbox in self.fast_box:
            speed = 1

        self.x_move *= speed
        self.y_move *= speed
        self.z_move *= speed

    def run(self):
        stream = VideoStream(self.stream_path)
        
        try:
            while True:
                # wait 
                if not self.tracked:
                    self.tracked, self.descriptor, position = self.get_info()
                    #self.client.goto(position)
                    continue

                frame = stream.read()

                if frame is not None:
                    
                    frame = cv2.rectangle(frame, self.stop_box.xyxy[:2], self.stop_box.xyxy[2:], (0, 0, 245), 5)
                    frame = self.preprocess(frame)

                    results = self.model.track(frame, device="cuda:0")

                    for res in results[0]:    
                        if res.boxes.cls != 1: # 1 is Drone
                            continue
                        
                        # update
                        if self.actual_id is None:
                            self.update_id(frame, res)
                        else:
                            self.update_moves(res)
                            break

                    else:
                        self.x_move = 0
                        self.y_move = 0
                        self.z_move = 0

                    # --- move ---
                    print(self.x_move, self.y_move, self.z_move,)
                    self.client.continuous_move(int(self.x_move), int(self.y_move), int(self.z_move))

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