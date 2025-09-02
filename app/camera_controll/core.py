import time
from multiprocessing import Process
import logging

from ultralytics import YOLO
import numpy as np
import zmq

from sources import VideoStream, coord_to_angle
from tracked_obj import TrackObject
from configs import AICoreConfig, ConnactionsConfig

class AICore(Process):
    def __init__(self, config_path=None, daemon = None):
        super().__init__(daemon=daemon)
        # TODO add configs and change other
        self.ai_config = AICoreConfig(config_path)
        self.state = "overview" # "standby", "tracking"
        self._overview_timout = 1
        self._standby_timeout = 3

        self.cameras = []
        self._track_index = None

        self.target = None
        self._no_target_count = 0
        self._no_target_limit = 25
        self._time_target_limit = 5

        self.detector = YOLO(self.ai_config.MODEL_PATH, task="detect")

    def _init_connaction(self):
        """Initialization socket for processes connaction.
        """

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.setsockopt(zmq.RCVHWM, 10)
        self.socket.bind(f"tcp://127.0.0.1:8000")

    def add_ip_camera(self, login, password, camera_ip, camera_port, for_track=False):
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        path = template.format(login, password, camera_ip, camera_port)
        stream = VideoStream(path)
        self.cameras.append(stream)

        if for_track:
            self._track_index = len(self.cameras) - 1

    def get_overview_frames(self) -> list:
        frames = []
        for i in len(self.cameras):
            if i != self._track_index:
                frame = self.cameras[i].read()
                frames.append(frame)

        return frames
    
    def get_tracking_frame(self) -> np.ndarray:
        if self._track_index is not None:
            return self.cameras[self._track_index].read()

    def get_biggest_info(self, detection_results):
        
        max_area = 0
        biggest_info = ()

        for camera_index, camera_results in enumerate(detection_results):
            if camera_results is None:
                continue

            for result in detection_results[0]:    
                if result.boxes.cls != self.t_config.DRONE_CLASS_ID:
                    continue

                x, y, w, h = result.boxes.xywh[0]
                obj_area = w * h

                if obj_area >= max_area:
                    max_area = obj_area

                    biggest_info = (camera_index, result.boxes.xywh[0])

        return biggest_info

    def send_target(self) -> bool:
        if self.target is not None:
            message = self.target.__dict__

            self.socket.send_json(message)

    def reset(self):
        self.target = None
        self._no_target_count = 0

    def overview(self) -> None:
        while self.target is None:
            frames = self.get_overview_frames()

            detection_results = []
            for frame in frames:
                result = self.model.predict(
                    frame, 
                    imgsz=(576, 1024),
                    conf=self.ov_config.DETECTOR_CONF,
                    iou=self.ov_config.DETECTOR_IOU,
                    )
                detection_results.append(result[0])

            info = self.get_biggest_info(detection_results)

            if info:
                index, bbox = info
                x, y = bbox[:2]

                x_calib = self.calibr_config.ALL[index]
                y_calib = self.ov_config.HORIZONT

                width = self.ov_config.WIDTH_FRAME
                hor = self.ov_config.HORIZ_ANGLE
                height = self.ov_config.HEIGHT_FRAME
                vert = self.ov_config.VERTIC_ANGLE

                abs_x = float(x_calib + coord_to_angle(x, width, hor))
                abs_y = float(y_calib - coord_to_angle(y, height, vert))

                relation = (0, 0) # default by overview
                absolute = (abs_x, abs_y)

                self.target = TrackObject(relation, absolute, bbox)

                self.state = "standby"
                self.send_target()
            else:
                time.sleep(self._overview_timout)

    def standby(self) -> TrackObject:
        pass

    def tracking(self) -> TrackObject:
        pass


    
    def run(self):
        logging.info("System initialization...")
        self._init_connaction()


        while self.running:
            logging.info(f"System state: {self.state}")

            if self.state == "overview":
                self.overview()
            elif self.state == "standby":
                self.standby()
            elif self.state == "tracking":
                self.tracking()
