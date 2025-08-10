from multiprocessing import Process, shared_memory
import json
import time
import logging

import cv2
from ultralytics import YOLO
import torch
import torch.nn.functional as F

from sources import BBox, descriptor
from sources import VideoStream
from sources import coord_to_steps, coord_to_angle
from sources import CarriageController
from configs import ConnactionsConfig, TrackerConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug descriptor for testing
DEBUG_DESCRIPTOR = descriptor(cv2.imread("./example.jpg"))

# Constants
DEFAULT_IMAGE_SIZE = 512

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
        self.tracked = False
        self.target_descriptor = None
        self.current_target_id = None
        self.last_found_time = None
        self.similarity_threshold = self.t_config.SIMILARITY_THRESHOLD

        # shared memory informations
        self.memory_name: str = "object_data" 
        self.shared_memory = None
        self._last_data_hash = None

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
    

    @staticmethod
    def preprocess_frame(frame, flip_code=0, target_size=DEFAULT_IMAGE_SIZE):
        """
        Preprocesses a video frame for model inference.

        This static method flips and resizes the frame to the target dimensions required by the model.

        Args:
            frame (numpy.ndarray): The input video frame.
            flip_code (int, optional): The flip code for `cv2.flip`. 
                                       0 for vertical, 1 for horizontal, -1 for both. Defaults to 0.
            target_size (int, optional): The target size for resizing. Defaults to DEFAULT_IMAGE_SIZE.

        Returns:
            numpy.ndarray: The preprocessed frame.
        """
        if frame is None:
            return None
            
        flipped_frame = cv2.flip(frame, flip_code)
        resized_frame = cv2.resize(flipped_frame, (target_size, target_size))
        return resized_frame

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

        # Create shared memory
        if self.shared_memory is None:
            self.shared_memory = shared_memory.SharedMemory(name=self.memory_name)

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
        
    def cleanup(self):
        """
        Cleans up resources by closing the shared memory connection.
        It does NOT unlink the memory, as the sender is responsible for that.
        """
        if self.shared_memory is not None:
            try:
                self.shared_memory.close()
                logging.info(f"Disconnected from shared memory '{self.memory_name}'.")
            except Exception as e:
                print(f"Error during receiver cleanup: {e}")

            finally:
                self.shared_memory = None
                
    def update_target_id(self, frame, detection_result):
        """
        Updates the current target ID by comparing descriptor similarity.

        This method computes the cosine similarity between the target descriptor and the
        descriptor of the detected object. If the similarity is above the threshold,
        the target ID is updated.

        Args:
            frame (numpy.ndarray): The current video frame.
            detection_result: The YOLO detection result for a potential target.

        Returns:
            bool: True if the target ID was successfully updated, False otherwise.
        """
        x1, y1, x2, y2 = map(int, detection_result.boxes.xyxy[0])
        object_region = frame[y1:y2, x1:x2]

        if object_region.size == 0 or detection_result.boxes.id is None:
            return False

        object_descriptor = descriptor(object_region)
    
        similarity = F.cosine_similarity(
            self.target_descriptor, 
            object_descriptor, 
            dim=0
        ).item()

        if similarity > self.similarity_threshold:
            self.current_target_id = detection_result.boxes.id[0].item()
            logger.info(f"Target updated with similarity: {similarity:.3f}")
            return True

        else:
            self.current_target_id = None
            return False

    def calculate_movement(self, object_bbox: BBox):
        """
        Calculates the required movement commands based on the target's position.

        If the target is within the central "stop area," no movement is generated.
        Otherwise, it calculates the necessary steps and angles to move the carriage.

        Args:
            object_bbox (BBox): The bounding box of the target object.
        """

        x, y = object_bbox[:2]

        width = self.t_config.WIDTH_FRAME
        hor = self.t_config.HORIZ_ANGLE
        height = self.t_config.HEIGHT_FRAME
        vert = self.t_config.VERTIC_ANGLE

        if object_bbox in self.stop_area:
            speed = 0.0
        else:
            speed = 1

        self.movement_x = int(coord_to_steps(x, width, hor) * speed)
        self.movement_y = -1 * int(coord_to_angle(y, height, vert) * speed) # -1 

        logger.debug(f"Movement calculated: X={self.movement_x}, Y={self.movement_y}")

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

    def detect_val(self, detection_results, frame) -> BBox:
        """
        Validates detection results to identify the tracked drone.

        This method iterates through detection results, filters for the correct class ID,
        and updates the target ID if a match is found based on descriptor similarity.

        Args:
            detection_results: The YOLO detection results.
            frame (numpy.ndarray): The current video frame.

        Returns:
            tuple: A tuple containing:
                   - bool: True if the target was found, False otherwise.
                   - BBox or None: The bounding box of the target if found.
        """
        target_found = False
        target_bbox = None

        for result in detection_results[0]:    
            if result.boxes.cls != self.t_config.DRONE_CLASS_ID:
                continue

            if self.current_target_id is None:
                if self.update_target_id(frame, result):
                    target_found = True
            elif self.current_target_id == result.boxes.id[0]:
                target_found = True
            else:
                continue
        
            target_bbox = BBox(*result.boxes.xywh[0])
        
        return target_found, target_bbox


    def run(self):
        """The main tracking loop of the process."""
        stream = VideoStream(self.stream_path)
        self.is_running = True
        
        try:
            logger.info("Starting carriage tracker...")
            while self.is_running:
                
                # wait message
                if self.tracked is False:
                    self.tracked, self.target_descriptor, position = self.get_tracking_info()
                    if position is not None:
                        self.controller.move_to_absolute(*position)
                    continue

                # get video frame
                frame = stream.read()

                if frame is None:
                    logger.warning("No frame received")
                    continue
                    
                try:
                    detection_results = self.model.track(
                        frame,
                        conf=self.t_config.DETECTOR_CONF,
                        iou=self.t_config.DETECTOR_IOU,
                        device="cuda:0")
                    
                    target_found, obj_bbox = self.detect_val(detection_results, frame)

                    if target_found:
                        self.last_found_time = time.time()
                        self.calculate_movement(obj_bbox)
                    else:
                        self.movement_x = 0
                        self.movement_y = 0
                        
                        is_found = self.last_found_time is not None
                        time_cond = time.time() - self.last_found_time > 5
                        
                        if is_found and time_cond:
                            self.tracked = False
                            self.last_found_time = None


                    if self.movement_x != 0 or self.movement_y != 0:
                        self.controller.move_relative(self.movement_x, self.movement_y)

                except Exception as e:
                    logger.error(f"Error in detection: {e}")
                    continue

                # Debug visualization (uncomment for debugging)
                # debug_frame = self._draw_debug_info(processed_frame.copy())
                # cv2.imshow("Carriage Tracker", debug_frame)
                
                # Check for exit condition
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Exit requested")
                    break

        except KeyboardInterrupt:
            logger.info("Tracking interrupted by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in tracking loop: {e}")
        finally:
            logger.info("Cleaning up...")
            self.cleanup()
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