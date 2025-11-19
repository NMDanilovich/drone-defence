"""
This module contains the main AI core for drone detection and tracking.
"""
import time
from multiprocessing import Process

from ultralytics import YOLO
import numpy as np
import zmq

from sources.logs import get_logger
from sources import VideoStream, ncoord_to_angle
from sources.tracked_obj import TrackObject
from configs import SystemConfig, ConnectionsConfig

logger = get_logger("Core_serv")

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
        self.config = SystemConfig()
        self.connections = ConnectionsConfig()

        self.state = "overview" # "standby", "tracking"
        self._overview_timeout = self.config.OVERVIEW["timeout"]
        self._standby_timeout = self.config.STANDBY["timeout"]

        self.cameras = []
        self._track_index = None

        self.target = None
        self._short_duration = 5
        self._long_duration = 10

        self.gst = True
        self.detector = YOLO(self.config.MODEL["path"], task="detect", verbose=True)
        self.image_size = self.config.MODEL["image_size"]

        self.running = False

    def _init_connection(self):
        """Initialization socket for processes connection.
        """

        try:
            self.context = zmq.Context.instance()

            self.context.socket(zmq.PUB)

            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.setsockopt(zmq.SNDHWM, 10)
            self.publisher.setsockopt(zmq.RCVHWM, 10)
            self.publisher.bind(f"tcp://127.0.0.1:8000")
        
        except zmq.error.ZMQError as error:
            logger.warning(f"Service connection error {error}")
            return False
        else:
            logger.info("Service connection is ready!")
            return True


    def _init_cameras(self):
        """Initializes the video streams from the cameras specified in the config."""
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"

        for i,  camera in enumerate(self.connections.data.values()):
            
            if camera["path"]:
                path = camera["path"]
            else:
                login = camera["login"]
                password = camera["password"]
                ip = camera["ip"]
                port = camera["port"]
                path = template.format(login, password, ip, port)
            
            stream = VideoStream(path, gst=self.gst, out_frame=self.image_size)

            if camera["track"]:
                self._track_index = i
                stream.stop()
                del stream
                stream = VideoStream(path, gst=self.gst, in_frame=(2560, 1440), fps=30, out_frame=self.image_size)

            self.cameras.append(stream)
        
        # Checking the connection to the cameras
        if len(self.cameras) != 0 and self._track_index is not None:
            logger.info("Cameras is ready!")
            return True
        else:
            logger.warning("The number of connected cameras is 0, or the tracking camera is not specified.")
            return False

    def _warmap_model(self):
        """Warmap YOLO for fastest inference"""
        
        try:
            for _ in range(10):
                dummy_input = np.random.randn(*self.image_size, 3)
                _ = self.detector.predict(
                        dummy_input, 
                        imgsz=self.image_size,
                        verbose=False
                        )
            
        except Exception as error:
            logger.warning(f"Undefined detector error {error}")
            return False
        else:
            logger.info("Detector is ready!")
            return True

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

                    biggest_info = [camera_index, result.boxes.xywhn[0]]

        if biggest_info:
            biggest_info[1] = biggest_info[1].cpu().tolist()

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

        hor = self.config.OVERVIEW["horiz_angle"]
        vert = self.config.OVERVIEW["vertic_angle"]

        x_angles = ncoord_to_angle(x, hor)
        y_angles = ncoord_to_angle(y, vert)

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
                    )
                detection_results.append(result[0])

            info = self.get_biggest_info(detection_results)

            if info:
                camera_index, bbox = info

                x_calib = self.config.CALIBRATION[f"camera_{camera_index}"]
                y_calib = self.config.OVERVIEW["horizont"]

                rel_x, rel_y = self.get_angles(bbox)

                abs_x = float(x_calib + rel_x)
                abs_y = float(y_calib - rel_y)

                absolute = (abs_x, abs_y)

                # initializate target
                self.target = TrackObject(camera_index, absolute, bbox, time=time.time())
                logger.info(f"Target initialization")
                
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
                    self.target.update(abs=absolute, box=bbox, tracked=False, error=(None, None))
                    logger.info(f"Update target from cameras {index}")
                else:
                    logger.info(f"Waiting for target updates")

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
                    self.target.update(error=(float(err_x), float(err_y)), tracked=True, box=bbox)
                    logger.info(f"Update target from tracking camera")
            
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

        if self.target is None:
            self.target = TrackObject(0, (0, 119), (0, 0, 20, 20), time=time.time())

        while time.time() - self.target.time < self._short_duration:
            frame = self.get_tracking_frame()

            if frame is None:
                continue

            detection_results = self.detector.track(
                frame, 
                imgsz=self.image_size,
                conf=self.config.MODEL["tracking_conf"],
                iou=self.config.MODEL["tracking_iou"],
                # verbose=True
                )
            
            detector_message = (
                "Tracking info: "
                f"Num objects: {len(detection_results[0])} "
                f"{detection_results}"
            )

            logger.info(detector_message)
            info = self.get_biggest_info(detection_results)
            # print(detection_results[0].boxes)

            if info:
                _, bbox = info

                err_x, err_y = self.get_angles(bbox)

                # update target
                self.target.update(error=(float(err_x), float(err_y)), box=bbox, tracked=True)

                self.send_target()
        
        self.state = "standby"

    def run(self):
        """
        The main loop of the AICore process.

        Initializes the system and then enters a loop to execute the logic
        for the current state (overview, standby, or tracking).
        """
        logger.info("System initialization...")
        self._init_connection()
        self._init_cameras()
        self._warmap_model()
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
            logger.error("Keyboard exit.")
        
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
