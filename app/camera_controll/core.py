"""
This module contains the main AI core for drone detection and tracking.
"""
import time
from multiprocessing import Process
import logging

from ultralytics import YOLO
import numpy as np
import zmq

from sources import VideoStream, coord_to_angle
from sources.tracked_obj import TrackObject
from configs import SystemConfig, ConnactionsConfig

logger = logging.getLogger("CoreServise")
logger.setLevel(logging.DEBUG)

class AICore(Process):
    """
    The main AI core process for drone detection and tracking.

    This class manages the state of the system (overview, standby, tracking),
    handles video streams from multiple cameras, runs object detection using YOLO,
    and communicates tracking information to other processes.
    """
    def __init__(self, daemon = None):
        """
        Initializes the AICore process.

        Args:
            daemon (bool, optional): Whether the process is a daemon. Defaults to None.
        """
        super().__init__(daemon=daemon)
        # TODO change configs
        self.config = SystemConfig()
        self.connactions = ConnactionsConfig()

        self.state = "overview" # "standby", "tracking"
        self._overview_timeout = self.config.OVERVIEW["timeout"]
        self._standby_timeout = self.config.STANDBY["timeout"]

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
        """Initializes the video streams from the cameras specified in the config."""
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"

        for i,  camera in enumerate(self.connactions.data.values()):
            
            if camera["path"]:
                path = camera["path"]
            else:
                login = camera["login"]
                password = camera["password"]
                ip = camera["ip"]
                port = camera["port"]
                path = template.format(login, password, ip, port)
            
            stream = VideoStream(path)
            self.cameras.append(stream)

            if camera["track"]:
                self._track_index = i

    def get_overview_frames(self) -> list:
        """
        Reads and returns frames from all overview cameras.

        Returns:
            list: A list of frames from the overview cameras.
        """
        frames = []
        for i, camera in enumerate(self.cameras):
            if i != self._track_index:
                frame = camera.read()
                frames.append(frame)

        return frames
    
    def get_tracking_frame(self) -> np.ndarray:
        """
        Reads and returns a frame from the tracking camera.

        Returns:
            np.ndarray: The frame from the tracking camera.
        """
        if self._track_index is not None:
            return self.cameras[self._track_index].read()

    def get_biggest_info(self, detection_results):
        """
        Finds the biggest detected drone from the detection results.

        Args:
            detection_results (list): A list of detection results from YOLO.

        Returns:
            tuple: A tuple containing the camera index and the bounding box
                   of the biggest detected drone. Returns an empty tuple if no
                   drone is found.
        """
        
        max_area = 0
        biggest_info = ()

        for camera_index, camera_results in enumerate(detection_results):
            if camera_results is None:
                continue

            for result in camera_results:    
                if result.boxes.cls != self.config.MODEL["drone_class_id"]:
                    continue

                x, y, w, h = result.boxes.xywh[0]
                obj_area = w * h

                if obj_area >= max_area:
                    max_area = obj_area

                    biggest_info = (camera_index, result.boxes.xywh[0])

        return biggest_info

    def get_angles(self, bbox):
        """
        Calculates the horizontal and vertical angles from the bounding box center.

        Args:
            bbox (list or tuple): The bounding box coordinates (x, y, w, h).

        Returns:
            tuple: A tuple containing the horizontal and vertical angles in degrees.
        """
        x, y = bbox[:2]

        width = self.config.OVERVIEW["width_frame"]
        hor = self.config.OVERVIEW["horiz_angle"]
        height = self.config.OVERVIEW["height_frame"]
        vert = self.config.OVERVIEW["vertic_angle"]

        x_angles = coord_to_angle(x, width, hor)
        y_angles = coord_to_angle(y, height, vert)

        return x_angles, y_angles

    def send_target(self) -> bool:
        """
        Sends the current target information via the publisher socket.
        """

        if self.target is not None:
            message = self.target.__dict__

            self.publisher.send_json(message)

    def reset(self):
        """Resets the current target."""
        self.target = None

    def overview(self) -> None:
        """
        The overview state logic.

        In this state, the system scans for drones using the overview cameras.
        If a drone is detected, it initializes a target and transitions to the
        standby state.
        """
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

                x_calib = self.config.CALIBRATION[f"camera_{index}"]
                y_calib = self.config.OVERVIEW["horizont"]

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
                time.sleep(self._overview_timeout)

    def standby(self) -> TrackObject:
        """
        The standby state logic.

        In this state, the system attempts to acquire a lock on the target
        using the tracking camera. If successful, it transitions to the tracking
        state. If the target is lost, it may return to the overview state.
        """
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

                    x_calib = self.config.CALIBRATION[f"camera_{index}"]
                    y_calib = self.config.OVERVIEW["horizont"]

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

                detection_results = self.detector.predict(
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
        """
        The tracking state logic.

        In this state, the system continuously tracks the target using the
        tracking camera and updates its position. If the target is lost for a
        certain duration, it transitions back to the standby state.
        """
        mean_time = 0

        while time.time() - self.target.time < self._short_duration:
            time_iter = time.time()

            time_getting = time.time()
            frame = self.get_tracking_frame()
            time_getting = time.time() - time_getting

            if frame is None:
                continue
            
            time_nn = time.time()
            detection_results = self.detector.predict(
                frame, 
                imgsz=self.image_size,
                conf=self.config.MODEL["tracking_conf"],
                iou=self.config.MODEL["tracking_iou"],
                verbose=False
                )
            time_nn = time.time() - time_nn

            handling_time = time.time()
            info = self.get_biggest_info(detection_results)

            if info:
                _, bbox = info

                err_x, err_y = self.get_angles(bbox)

                # update target
                self.target.update(error=(float(err_x), float(err_y)), box=bbox.cpu().tolist())

                self.send_target()
            handling_time = time.time() - handling_time
            
            time_iter = time.time() - time_iter

            print(f"durations: \n\titeration: {time_iter} \n\tframe getting: {time_getting} \n\tyolo: {time_nn} \n\thandling: {handling_time}")

            if mean_time:
                mean_time += time_nn
                mean_time /= 2
            else:
                mean_time = time_nn

            print(round(mean_time, 3) * 1000, "ms")

        self.state = "standby"

    def run(self):
        """
        The main loop of the AICore process.

        Initializes the system and then enters a loop to execute the logic
        for the current state (overview, standby, or tracking).
        """
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
    """
    The main function to start the AICore process.
    """
    core = AICore()
    core.start()

if __name__ == "__main__":
    main()
