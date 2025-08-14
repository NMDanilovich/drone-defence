from dataclasses import dataclass
from multiprocessing import Process, shared_memory
import uuid
import json
import time
import logging

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F

from sources import BBox, descriptor
from sources import VideoStream
from sources import coord_to_steps, coord_to_angle
from sources import CarriageController
from configs import ConnactionsConfig, TrackerConfig


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Debug descriptor for testing
DEBUG_DESCRIPTOR = descriptor(cv2.imread("./example.jpg"))

# Constants
DEFAULT_IMAGE_SIZE = 512

@dataclass
class TrackingObject:
    id: int = uuid.uuid4()
    id_class: int 
    descriptor: torch.Tensor
    rel_coord: tuple
    abs_coord: tuple
    xywh: tuple = None
    tracked: bool = False
    found_time: float = time.time()


class Tracker(Process):
    """
    A process-based drone tracker that controls the movement of a camera carriage.

    This class is designed to run as a separate process for tracking drones in a video stream.
    It receives tracking information from an overview module via shared memory, identifies the target
    drone using a YOLO model, and sends movement commands to a carriage controller to keep the
    drone centered in the frame.

    Args:
        login (str): The login username for the RTSP camera stream.
        password (str): The password for the RTSP camera stream.
        camera_ip (str): The IP address of the camera.
        camera_port (int): The port for the camera stream.
        config_path (str, optional): Path to the tracker configuration file. Defaults to None.

    Attributes:
        t_config (TrackerConfig): Configuration settings for the tracker.
        stream_path (str): The URL for the RTSP camera stream.
        frame_width (int): The width of the video frames.
        frame_height (int): The height of the video frames.
        model (YOLO): The YOLO object detection model.
        controller (CarriageController): The controller for the camera carriage.
        stop_area (BBox): A bounding box defining the central "stop" area where no movement is needed.
        tracked (bool): A flag indicating if a drone is currently being tracked.
        target_descriptor (torch.Tensor): The feature descriptor of the target drone.
        current_target_id (int): The ID of the currently tracked drone.
        last_found_time (float): The timestamp when the drone was last detected.
        similarity_threshold (float): The threshold for cosine similarity to match descriptors.
        shared_memory (shared_memory.SharedMemory): The shared memory object for communication.
    """
    def __init__(self, login, password, camera_ip, camera_port, config_path=None):
        super().__init__()
        self.t_config = TrackerConfig(config_path)
        
        # stream configuration
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        self.stream_path = template.format(login, password, camera_ip, camera_port)

        # get camera settings
        self.frame_width = self.t_config.WIDTH_FRAME
        self.frame_height = self.t_config.HEIGHT_FRAME
        self.frame_center = self.frame_width // 2, self.frame_height // 2

        # initialize AI model and UART controller
        self.model = YOLO(self.t_config.MODEL_PATH, task="detect")
        self.controller = CarriageController()
        # self.controller.move_to_start()
        self.reset_timeout = 5 # sec

        # Carriage moves
        self.movement_x = 0
        self.movement_y = 0
        self.movement_gain = 0.5

        # drone information
        self.target_drone: TrackingObject = None

        self.similarity_threshold = self.t_config.SIMILARITY_THRESHOLD

        # shared memory informations
        self.memory_name: str = "object_data" 
        try:
            self.shared_memory = self.shared_memory = shared_memory.SharedMemory(name=self.memory_name)
            self.single_mode = False
        except FileNotFoundError as err:
            logger.warning("Shered memmory not found. Single mode ON.")
            self.shared_memory = None
            self.single_mode = True

        try:
            self.shared_memory = self.shared_memory = shared_memory.SharedMemory(name=self.memory_name)
            self.single_mode = False
        except FileNotFoundError as err:
            logger.warning("Shered memmory not found. Single mode ON.")
            self.shared_memory = None
            self.single_mode = True

        self._last_data_hash = None

    def init_target_drone(self):

        self.target_drone = None

        tracked, descriptor, position = self.get_tracking_info()
        if tracked:
            self.target_drone = TrackingObject(
                id_class=self.t_config.DRONE_CLASS_ID,
                descriptor=descriptor,
                abs_coord=position,
                tracked=False
            )

        return self.target_drone

    def get_tracking_info(self):
        """
        Retrieves tracking information from the overview module via shared memory.

        This method reads data from the shared memory, deserializes it, and checks if it's new data
        by comparing hashes.

        Returns:
            tuple: A tuple containing:
                   - bool: True if tracking information is available and new, False otherwise.
                   - torch.Tensor or None: The feature descriptor of the target.
                   - tuple or None: The (x, y) position of the target.
        """

        try:
            # read the bytes from shared memory
            raw_bytes = self.shared_memory.buf[:]

            received_json = bytes(raw_bytes).decode('utf-8').strip('\x00')  # Remove padding
            
            current_hash = hash(received_json)
            if current_hash == self._last_data_hash:
                return False, None, None
            else:
                self._last_data_hash = current_hash       
            
            if received_json == "":
                return False, None, None

            # get object information
            data = json.loads(received_json)

            return True, torch.Tensor(data["object_descriptor"]), (data["x_position"], data["y_position"])

        except Exception as error:
            print("\rError getting object information:", error, end="")
            return False, None, None
        
    def update_target_drone(self) -> BBox:
        
        detection_results = self.model.predict(
            self.frame,
            conf=self.t_config.DETECTOR_CONF,
            iou=self.t_config.DETECTOR_IOU,
            device="cuda:0"
        )
        
        max_area = 0
        

        for result in detection_results[0]:    
            if result.boxes.cls != self.t_config.DRONE_CLASS_ID:
                continue

            x, y, w, h = result.boxes.xywh[0]
            obj_area = w * h

            if obj_area >= max_area:
                max_area = obj_area

                x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
                object_region = self.frame[y1:y2, x1:x2]

                desc = descriptor(object_region)

                rel_coord = self.calc_position(x, y)
                carriage_pos = self.controller.get_position()
                abs_coord = carriage_pos[0] + rel_coord[0], carriage_pos[1] + rel_coord[1]

                self.target_drone.descriptor = desc
                self.target_drone.rel_coord = rel_coord
                self.target_drone.abs_coord = abs_coord
                self.target_drone.found_time = time.time()
                self.target_drone.tracked = True
                break
        else:
            self.target_drone = None

        return self.target_drone
    
    def calc_position(self, x: int, y: int) -> tuple:

        width = self.t_config.WIDTH_FRAME
        hor = self.t_config.HORIZ_ANGLE
        height = self.t_config.HEIGHT_FRAME
        vert = self.t_config.VERTIC_ANGLE

        x_pos = int(coord_to_steps(x, width, hor))
        y_pos = -1 * int(coord_to_angle(y, height, vert)) # -1 is reverce. Need to engine coordinates 

        logger.debug(f"Position calculated: X={x_pos}, Y={y_pos}")

        return x_pos, y_pos

    def calc_movement(self, x, y):

        width = self.t_config.WIDTH_FRAME
        hor = self.t_config.HORIZ_ANGLE
        height = self.t_config.HEIGHT_FRAME
        vert = self.t_config.VERTIC_ANGLE
        
        norm_x = (x - width / 2) / (width / 2)
        norm_y = (y - height / 2) / (height / 2)
        
        speed_x = abs(norm_x) * 0.3
        speed_y = abs(norm_y) * 0.3
        
        movement_x = int(coord_to_steps(x, width, hor) * speed_x)
        movement_y = -1 * int(coord_to_angle(y, height, vert) * speed_y)
        
        logger.debug(f"Movement calculated: X={self.movement_x}, Y={self.movement_y}")

        return movement_x, movement_y

    def calc_time(self, x_steps, y_angles):
        x_speed = self.controller.x_speed * self.controller.x_spd
        y_speed = self.controller.y_speed

        x_time = x_steps / x_speed
        y_time = y_angles / y_speed

        return x_time, y_time

    @staticmethod
    def calc_dist(point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        dist = np.linalg.norm(point1 - point2)
        return dist

    def get_move(self) -> dict:
        """Getting move to object information

        Returns:
            dict: object information has a structure:
        ```python
        {
            "next_move": [tuple[int, int]]  # move to position for engine
            "distance": [int]  # pixels between drone bbox center and frame center
            "time_move": [float]  # time for engine move
        }
        ```
        """
        x_move, y_move = self.target_drone.rel_coord
        next_move = int(x_move * self.movement_gain), int(y_move * self.movement_gain)
        time = self.calc_time(*next_move)
        drone_center = self.target_drone.xywh[:2]
        distance = self.calc_dist(self.frame_center, drone_center)

        return {"next_move": next_move, "distance": distance, "time_move": time}

    def cleanup(self):
        """
        Cleans up resources by closing the shared memory connection.
        It does NOT unlink the memory, as the sender is responsible for that.
        """
        if self.shared_memory is not None:
            try:
                self.shared_memory.close()
                logger.info(f"Disconnected from shared memory '{self.memory_name}'.")
            except Exception as e:
                print(f"Error during receiver cleanup: {e}")

            finally:
                self.shared_memory = None
                

    def _draw_debug_info(self, frame):
        """Draws debugging information on the frame."""
        # Draw stop area
        cv2.rectangle(
            frame, 
            self.stop_area.xyxy[:2], 
            self.stop_area.xyxy[2:], 
            (0, 255, 0), 
            2
        )
        
        # Add status text
        status_text = f"Tracking: {self.is_tracking}, ID: {self.current_target_id}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def run(self):
        """The main tracking loop of the process."""
        stream = VideoStream(self.stream_path)
        self.is_running = True

        try:
            logger.info("Starting carriage tracker...")
            while self.is_running:
                

                if not self.single_mode:
                    while self.target_drone is None:
                        self.init_target_drone()

                    self.controller.move_to_absolute(*self.target_drone.abs_coord)


                plan_move = None
                while True:
                    # get video frame
                    self.frame = stream.read()

                    if self.frame is None:
                        logger.warning("No frame received")
                        continue
                    
                    self.update_target_drone()

                    if self.target_drone is not None:
                        if plan_move is None:
                            # get plan
                            plan_move = self.get_move()
                            start_move_time = time.time()
                            self.controller.move_relative(*plan_move["next_move"])
                            # half step
                            self.movement_gain = 0.5

                        # actual state
                        cur_move = self.get_move()

                        # distances is equal
                        if np.allclose(plan_move["distance"], cur_move["distance"], rtol=5):
                            if time.time() - start_move_time >= plan_move["time_move"]:
                                self.movement_gain = 1 # full step
                                plan_move = None # reset plan to repeat movement

                        # target is done
                        elif np.allclose(cur_move["distance"], 0, rtol=5):
                            plan_move = None

                        elif cur_move["distance"] > plan_move["distance"]:
                            self.movement_gain = 1
                            plan_move = None

                        elif cur_move["distance"] < plan_move["distance"]:
                            if time.time() - start_move_time >= plan_move["time_move"]:
                                self.movement_gain = 1.5
                                plan_move = None                                


                    else:
                        logger.warning("No drone detected")
                        if time.time() - self.target_drone.found_time > self.reset_timeout:
                            logger.info("Reset system")
                            self.target_drone = None
                            break

                    #Debug visualization (uncomment for debugging)
                    # debug_frame = self._draw_debug_info(frame.copy())


        except KeyboardInterrupt:
            logger.info("Tracking interrupted by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in tracking loop: {e}")
        finally:
            logger.info("Cleaning up...")
            stream.stop()
            self.cleanup()
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
