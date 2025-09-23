import time
from multiprocessing import Process
import logging

from ultralytics import YOLO
import numpy as np
import zmq

from sources import VideoStream, coord_to_angle
from sources.tracked_obj import TrackObject
from configs import SystemConfig, OverviewConfig, TrackerConfig, ConnactionsConfig, CalibrationConfig

logger = logging.getLogger("CoreServise")
logger.setLevel(logging.DEBUG)

class AICore(Process):
    def __init__(self, daemon = None):
        super().__init__(daemon=daemon)
        # TODO change configs
        self.config = SystemConfig()
        self.conn_conf = ConnactionsConfig()
        self.ov_config = OverviewConfig()
        self.t_config = TrackerConfig()
        self.calibr_config = CalibrationConfig()

        self.state = "overview" # "standby", "tracking"
        self._overview_timout = 0.1
        self._standby_timeout = 1

        self.cameras = []
        self._track_index = None

        self.target = None
        self._short_duration = 5
        self._long_duration = 10

        self.detector = YOLO(self.config.MODEL["path"], task="detect")
        self.image_size = (576, 1024)
        self.running = False

    def _init_connaction(self):
        """Initialization socket for processes connaction.
        """

        self.context = zmq.Context.instance()

        self.context.socket(zmq.PUB)

        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.SNDHWM, 10)
        self.publisher.setsockopt(zmq.RCVHWM, 10)
        self.publisher.bind(f"tcp://127.0.0.1:8000")


    def _init_cameras(self):
        for camera in self.conn_conf.data:

            login = self.conn_conf.data[camera]["login"]
            password = self.conn_conf.data[camera]["password"]
            ip = self.conn_conf.data[camera]["ip"]
            port = self.conn_conf.data[camera]["port"]

            self.add_ip_camera(login, password, ip, port, for_track=self.conn_conf.data[camera]["track"])

    def add_ip_camera(self, login, password, camera_ip, camera_port, for_track=False):
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        path = template.format(login, password, camera_ip, camera_port)
        stream = VideoStream(path)
        self.cameras.append(stream)

        if for_track:
            self._track_index = len(self.cameras) - 1

    def get_overview_frames(self) -> list:
        frames = []
        for i, camera in enumerate(self.cameras):
            if i != self._track_index:
                frame = camera.read()
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

            for result in camera_results:    
                if result.boxes.cls != self.t_config.DRONE_CLASS_ID:
                    continue

                x, y, w, h = result.boxes.xywh[0]
                obj_area = w * h

                if obj_area >= max_area:
                    max_area = obj_area

                    biggest_info = (camera_index, result.boxes.xywh[0])

        return biggest_info

    def get_angles(self, bbox):
        x, y = bbox[:2]

        width = self.ov_config.WIDTH_FRAME
        hor = self.ov_config.HORIZ_ANGLE
        height = self.ov_config.HEIGHT_FRAME
        vert = self.ov_config.VERTIC_ANGLE

        x_angles = coord_to_angle(x, width, hor)
        y_angles = coord_to_angle(y, height, vert)

        return x_angles, y_angles

    def send_target(self) -> bool:

        if self.target is not None:
            message = self.target.__dict__

            self.publisher.send_json(message)

    def reset(self):
        self.target = None

    def overview(self) -> None:
        while self.target is None:
            frames = self.get_overview_frames()

            if not frames:
                continue

            detection_results = []
            for frame in frames:
                result = self.detector.predict(
                    frame, 
                    imgsz=self.image_size,
                    conf=self.config.MODEL["overview_conf"],
                    iou=self.config.MODEL["overview_iou"],
                    verbose=False
                    )
                detection_results.append(result[0])

            info = self.get_biggest_info(detection_results)

            if info:
                index, bbox = info

                x_calib = self.calibr_config.ALL[index]
                y_calib = self.ov_config.HORIZONT

                rel_x, rel_y = self.get_angles(bbox)

                abs_x = float(x_calib + rel_x)
                abs_y = float(y_calib - rel_y)

                absolute = (abs_x, abs_y)

                # initializate target
                self.target = TrackObject(absolute, bbox.cpu().tolist(), time=time.time())
                logger.info(f"Init target: {self.target}")
                
                self.state = "standby"
                self.send_target()
            else:
                logger.info(f"No drone information")
                time.sleep(self._overview_timout)

    def standby(self) -> TrackObject:
        tracked = False
        standby_time = time.time()

        while not tracked:
            if time.time() - standby_time > self._standby_timeout:
                frames = self.get_overview_frames()

                if not frames:
                    continue

                detection_results = []
                for frame in frames:
                    result = self.detector.predict(
                        frame, 
                        imgsz=self.image_size,
                        conf=self.config.MODEL["overview_conf"],
                        iou=self.config.MODEL["overview_iou"],
                        verbose=False
                        )
                    detection_results.append(result[0])
                
                standby_time = time.time()

                info = self.get_biggest_info(detection_results)

                if info:
                    index, bbox = info

                    x_calib = self.calibr_config.ALL[index]
                    y_calib = self.ov_config.HORIZONT

                    rel_x, rel_y = self.get_angles(bbox)

                    abs_x = float(x_calib + rel_x)
                    abs_y = float(y_calib - rel_y)

                    absolute = (abs_x, abs_y)

                    # initializate target
                    self.target.update(absolute, bbox.cpu().tolist(), tracked=False, error=(None, None))
                    logger.info(f"OV. Update target: {self.target}")

            else:

                frame = self.get_tracking_frame()

                if frame is None:
                    continue

                detection_results = self.detector.track(
                    frame, 
                    imgsz=self.image_size,
                    conf=self.config.MODEL["tracking_conf"],
                    iou=self.config.MODEL["tracking_iou"],
                    )
                
                info = self.get_biggest_info(detection_results)

                if info:
                    self.state = "tracking"
                    tracked = True

                    _, bbox = info
                    err_x, err_y = self.get_angles(bbox)
                    self.target.update(error=(float(err_x), float(err_y)), tracked=True, box=bbox.cpu().tolist())
                    logger.info(f"T. Update target: {self.target}")
            
            self.send_target()

            if time.time() - self.target.time >= self._long_duration:
                self.state = "overview"
                self.reset()
                break


    def tracking(self) -> TrackObject:
        while time.time() - self.target.time < self._short_duration:
            frame = self.get_tracking_frame()

            if frame is None:
                continue

            detection_results = self.detector.track(
                frame, 
                imgsz=self.image_size,
                conf=self.config.MODEL["tracking_conf"],
                iou=self.config.MODEL["tracking_iou"],
                verbose=False
                )

            info = self.get_biggest_info(detection_results)

            if info:
                _, bbox = info

                err_x, err_y = self.get_angles(bbox)

                # update target
                self.target.update(error=(float(err_x), float(err_y)), box=bbox.cpu().tolist())

                self.send_target()
        
        self.state = "standby"

    def run(self):
        logger.info("System initialization...")
        self._init_connaction()
        self._init_cameras()
        self.running = True
        
        try:
            while self.running:
                logger.info(f"System state: {self.state}")

                if self.state == "overview":
                    self.overview()
                elif self.state == "standby":
                    self.standby()
                elif self.state == "tracking":
                    self.tracking()
        except KeyboardInterrupt:
            self.running = False
            logger.exception("Keyboard exit.")
        
        finally:
            self.context.destroy()
            
            for camera in self.cameras:
                camera.stop()

def main():
    core = AICore()
    core.start()

if __name__ == "__main__":
    main()
