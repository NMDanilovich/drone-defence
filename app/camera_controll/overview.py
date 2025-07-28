import time
import logging

from ultralytics import YOLO

from stream import VideoStream
from uartapi import Uart
import config

class Overview:
    def __init__(self, login, password, cameras_ips, cameras_ports, model_path):
        self.cameras_ips = cameras_ips
        self.cameras_ports = cameras_ports
        self.num_cameras = len(cameras_ips)

        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        self.streams_path = [template.format(login, password, ip, port) for ip, port in zip(self.cameras_ips, self.cameras_ports)]

        self.model = YOLO(model_path, task="detect")


    def run(self):
        # run Videostream threads
        streams = [VideoStream(path) for path in self.streams_path]

        try:
            while True:
                start = time.time()

                # get video frames
                frames = []
                for stream in streams:
                    frame = stream.read()
                    if frame is not None:
                        frames.append(frame)
                
                # get bboxes
                if len(frames) == self.num_cameras:
                    results = self.model.predict(frames,  stream=True)

                # get nearest object
                nearest_obj = {
                    "camera": None,
                    "object": None,
                    "center": None
                }

                max_area = 0
                for camera_index, camera_results in enumerate(results):
                    if camera_results is None:
                        continue

                    for object_index, object in enumerate(camera_results):
                        if object.boxes.cls == 1:

                            *center, w, h = object.boxes.xywh
                            # x, y = center
                            object_area = w * h

                            if object_area > max_area:
                                max_area = object_area

                                nearest_obj["camera"] = camera_index
                                nearest_obj["object"] = object_index
                                nearest_obj["center"] = center

                # TODO send message on 

                print("Total time:", time.time() - start)

        finally:
            for stream in streams:
                stream.stop()

def main():
    """Main function for running process
    """
    ip_cameras_config = config.OVERVIEW
    model_path = config.MODEL_PATH

    login = ip_cameras_config["login"]
    password = ip_cameras_config["password"]
    cameras_ips = [ip_cameras_config[camera]["ip"] for camera in ip_cameras_config if camera.startswith("camera")]
    cameras_ports = [ip_cameras_config[camera]["port"] for camera in ip_cameras_config if camera.startswith("camera")]
    
    overview = Overview(login, password, cameras_ips, cameras_ports, model_path)
    overview.run()

if __name__ == "__main__":
    main()