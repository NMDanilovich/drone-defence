import time

import torch
from torchvision.transforms import v2
from ultralytics import YOLO
import cv2

from stream import VideoStream
import config

class Overview:
    def __init__(self, cameras_ip, login, password, cameras_port, model_path):
        self.cameras_ip = cameras_ip
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        self.streams_path = [template.format(login, password, ip, cameras_port) for ip in self.cameras_ip]
        self.num_cameras = len(cameras_ip)
        self.__model = YOLO(model_path, task="detect")
        self.transforms = v2.Compose([
            v2.ToTensor(),
            v2.Resize(512),
        ])

    def run(self):
        streams = [VideoStream(path) for path in self.streams_path]

        try:
            while True:
                start = time.time()

                frames = []
                for stream in streams:
                    frame = stream.read()
                    if frame is not None:
                        frames.append(frame)
                    
                
                if len(frames) == self.num_cameras:
                    results = self.__model.predict(frames,  stream=True)

                print("Total time:", time.time() - start)

        finally:
            for stream in streams:
                stream.stop()

if __name__ == "__main__":
    cameras_ip = config.CAMERAS_IP
    cameras_port = config.PORT
    login = config.LOGIN
    password = config.PASSWORD
    model_path = config.MODEL_PATH
    select_id = 1

    overview = Overview(cameras_ip, login, password, cameras_port, model_path)
    overview.run()