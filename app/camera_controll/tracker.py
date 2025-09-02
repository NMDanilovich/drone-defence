from multiprocessing import Process
from dataclasses import dataclass
import time
import logging

import cv2
import numpy as np
from ultralytics import YOLO
import zmq

from sources import BBox
from sources import VideoStream
from sources import coord_to_angle
from sources import CarriageController
from configs import ConnactionsConfig, TrackerConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_SIZE = 512

@dataclass
class TrackObject:
    rel: tuple
    abs: tuple
    box: tuple

    time: float = time.time()
    timeout = 15 # sec

    def update(self, rel=None, abs=None, box=None):
        self.rel = self.rel if rel is None else rel
        self.abs = self.abs if abs is None else abs
        self.box = self.box if box is None else box
        self.time = time.time()


class Tracker(Process):
    def __init__(self, login, password, camera_ip, camera_port, config_path=None):
        super().__init__()
        self.t_config = TrackerConfig(config_path)
        
        # stream configuration
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        self.stream_path = template.format(login, password, camera_ip, camera_port)

        # get camera settings
        self.frame_width = None
        self.frame_height = None
        self._init_camera()
        
        # initialize AI model and UART controller
        self.model = YOLO(self.t_config.MODEL_PATH, task="detect")
        self.controller = CarriageController()
        self.controller.move_to_start()

        # set image areas
        self._setup_tracking_areas()

        # Carriage moves
        self.movement_x = 0
        self.movement_y = 0

        # drone information
        self.target = None
        self.__frames_without_target = 0
        self.__max_no_target = 25

        self.contexts: zmq.Context = None 
        self.subscriber: zmq.Socket = None

    def _init_camera(self):
        cap = cv2.VideoCapture(self.stream_path)
        
        if not cap.isOpened():
            raise FileNotFoundError(f"Stream not found: {self.stream_path}")
            
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")
    
    def _setup_tracking_areas(self):
        """Sets up the tracking areas for movement control based on the frame dimensions."""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        
        # Stop area - when target is in this area, minimal movement
        stop_width = int(self.frame_width * self.t_config.STOP_AREA_SCALE)
        stop_height = int(self.frame_height * self.t_config.STOP_AREA_SCALE)
        
        self.stop_area = BBox(center_x, center_y, stop_width, stop_height)
        
        logger.info(f"Stop area configured: {self.stop_area}")

    def _setup_connaction(self, filter_msg:str=""):
        """Setting up connection to proxy
        
        Args:
            filter_msg (str): Filter for ZMQ listner messeges. Can be used for recive message from specific camera.
        """
        self.ctx = zmq.Context.instance()
        self.subscriber = self.ctx.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://127.0.0.1:8000")
        self.subscriber.subscribe(filter_msg)

    def init_target(self):
        data = self.subscriber.recv_json()
        position = data["x_position"], data["y_position"]
        self.target = TrackObject((0,0), position, (100, 100, 50, 50))
        return self.target

    def update_target(self) -> TrackObject:
        
        detection_results = self.model.predict(
            self.frame,
            imgsz=(576, 1024),
            conf=self.t_config.DETECTOR_CONF,
            iou=self.t_config.DETECTOR_IOU,
            device="cuda:0"
        )
        
        if len(detection_results[0]) == 0:
            self.__frames_without_target += 1

        max_area = 0
        
        for result in detection_results[0]:    
            if result.boxes.cls != self.t_config.DRONE_CLASS_ID:
                continue

            x, y, w, h = result.boxes.xywh[0]
            obj_area = w * h

            if obj_area >= max_area:
                self.__frames_without_target = 0
                max_area = obj_area

                rel_coord = self.calc_position(x, y)
                carriage_pos = self.controller.get_position()
                abs_coord = carriage_pos[0] + rel_coord[0], carriage_pos[1] + rel_coord[1]

                if self.target is None:
                    self.target = TrackObject(
                        rel=rel_coord,
                        abs=abs_coord,
                        box=result.boxes.xywh[0]
                    )
                else:
                    self.target.update(rel_coord, abs_coord, result.boxes.xywh[0])
        
        if self.__frames_without_target == self.__max_no_target:
            self.target = None

        return self.target
    

    def calc_position(self, x: int, y: int) -> tuple:

        width = self.t_config.WIDTH_FRAME
        hor = self.t_config.HORIZ_ANGLE
        height = self.t_config.HEIGHT_FRAME
        vert = self.t_config.VERTIC_ANGLE

        x_pos = float(1 * coord_to_angle(x, width, hor))
        y_pos = float(-1 * coord_to_angle(y, height, vert)) # -1 is reverce. Need to engine coordinates 

        logger.debug(f"Position calculated: X={x_pos}, Y={y_pos}")

        return x_pos, y_pos

    def run(self):
        """The main tracking loop of the process."""
        stream = VideoStream(self.stream_path)
        self._setup_connaction()
        self.is_running = True
        
        try:
            logger.info("Starting carriage tracker...")
            while self.is_running:
                
                self.init_target()
                    
                if self.target is not None:
                    self.controller.move_to_absolute(*self.target.abs)

                fire = False
                while self.target is not None:
                    self.frame = stream.read()
                    

                    if self.frame is None:
                        logger.warning("No frame received")
                        continue

                    self.update_target()
                    print(self.target)
                    if self.target is not None:
                        part = 4
                        part_move = self.target.rel[0] / part, self.target.rel[1] / part
                        self.controller.move_relative(*part_move)

                        # center = self.target.box[:2]
                        # stop_center = np.array(self.stop_area[:2])
                        # distance = np.linalg.norm(center - stop_center)

                        self.target.update(rel=(0, 0))

                        # TODO rewrite this
                        # trying to continuous movement or synchrone detection method (by Petr)
   

                        # if self.target.box[:2] in self.stop_area and not fire:
                        #     self.controller.fire("fire")
                        #     fire = True
                        # elif self.target.box[:2] not in self.stop_area and fire:
                        #     self.controller.fire("stop") 
                        #     fire = False 
                        
                        while not self.controller.uart.exec_status():
                            print('Waiting for command execution')
                            time.sleep(0.002)

        except KeyboardInterrupt:
            logger.info("Tracking interrupted by user")
            self.is_running = False
            
        # except Exception as e:
        #     logger.error(f"Unexpected error in tracking loop: {e}")
        finally:
            logger.info("Cleaning up...")
            self.controller.fire("stop")
            stream.stop()
            cv2.destroyAllWindows()

def main():
    """Initializes and runs the Tracker process for standalone execution."""

    connactions = ConnactionsConfig()
    login = connactions.T_CAMERA_1["login"]
    password = connactions.T_CAMERA_1["password"]
    ip = connactions.T_CAMERA_1["ip"]
    port = connactions.T_CAMERA_1["port"]

    tracker = Tracker(
        login=login,
        password=password,
        camera_ip=ip,
        camera_port=port,
    )

    tracker.start()
    tracker.join()

if __name__ == "__main__":
    main()
