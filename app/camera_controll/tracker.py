from multiprocessing import Process, shared_memory
import json
import logging

import cv2
from ultralytics import YOLO
import torch.nn.functional as F

from src.model_utils import BBox, descriptor
from src.stream import VideoStream
from src.uartapi import Uart, JETSON_SERIAL
import cfg.connactions as conn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug descriptor for testing
DEBUG_DESCRIPTOR = descriptor(cv2.imread("./test_drone.png"))

# Constants
DRONE_CLASS_ID = 1
DEFAULT_IMAGE_SIZE = 512
STOP_AREA_SCALE = 0.2
SIMILARITY_THRESHOLD = 0.7
MOVEMENT_MULTIPLIER = 2
COORDINATE_SCALE = 100

DETECTOR_CONF = 0.4
DETECTOR_IOU = 0.7

class Tracker(Process):
    """
    A process-based drone tracker that controls camera carriage movement.
    
    This class tracks drones in video stream and sends movement commands
    to control the camera carriage via UART communication.
    """
    def __init__(self, stream_path, model_path, controller_path=JETSON_SERIAL):
        super().__init__()

        # --- stream configuration ---
        self.stream_path = stream_path

        # --- get camera settings ---
        self.frame_width = None
        self.frame_height = None
        self._init_camera()
        
        # --- initialize AI model and UART controller ---
        self.model = YOLO(model_path, task="detect")
        self.uart = Uart(controller_path)

        # --- set image areas ---
        self._setup_tracking_areas()

        # --- Carriage moves ---
        self.movement_x = 0
        self.movement_y = 0
        self.zoom_level = 0

        # --- drone information ---
        self.is_tracking = False
        self.target_descriptor = None
        self.current_target_id = None
        self.similarity_threshold = SIMILARITY_THRESHOLD

        self.shared_memory = shared_memory.SharedMemory(name="object_data")

    def _init_camera(self):
        cap = cv2.VideoCapture(self.stream_path)
        
        if not cap.isOpened():
            raise FileNotFoundError(f"Stream not found: {self.stream_path}")
            
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")
    
    def _setup_tracking_areas(self):
        """Setup tracking areas for movement control."""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        
        # Stop area - when target is in this area, minimal movement
        stop_width = int(self.frame_width * STOP_AREA_SCALE)
        stop_height = int(self.frame_height * STOP_AREA_SCALE)
        
        self.stop_area = BBox(center_x, center_y, stop_width, stop_height)
        
        logger.info(f"Stop area configured: {self.stop_area}")
    

    @staticmethod
    def preprocess_frame(frame, flip_code=0, target_size=DEFAULT_IMAGE_SIZE):
        """
        Preprocess frame for model inference.
        
        Args:
            frame: Input frame
            flip_code: Flip code for cv2.flip (0=vertical, 1=horizontal, -1=both)
            target_size: Target size for resizing
            
        Returns:
            Preprocessed frame
        """
        if frame is None:
            return None
            
        flipped_frame = cv2.flip(frame, flip_code)
        resized_frame = cv2.resize(flipped_frame, (target_size, target_size))
        return resized_frame

    def get_tracking_info(self):
        """
        Get tracking information from overview module.
        
        Returns:
            tuple: (should_track, target_descriptor, target_position)
        """

        try:
            received_json = bytes(self.shared_memory.buf[:]).decode('utf-8').strip('\x00')  # Remove padding
            data = json.loads(received_json)

            return True, data["descriptor"], (data["x_position"], data["y_position"])

        except Exception as error:
            print("Get object iformation error:", error)
            return False, None, None

    def update_target_id(self, frame, detection_result):
        """
        Update the current target ID based on descriptor similarity.
        
        Args:
            frame: Current frame
            detection_result: YOLO detection result
            
        Returns:
            bool: True if target ID was successfully updated
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

    def calculate_movement(self, detection_result):
        """
        Calculate movement commands based on target position.
        
        Args:
            detection_result: YOLO detection result
        """

        target_bbox = BBox(*detection_result.boxes.xywh[0])
        
        if self.current_target_id == detection_result.boxes.id:
            
            center_offset_x = self.stop_area[0] - target_bbox[0]
            center_offset_y = self.stop_area[1] - target_bbox[1]
            
            self.movement_x = int(
                    center_offset_x * -MOVEMENT_MULTIPLIER * COORDINATE_SCALE // self.frame_width
                )
            
            self.movement_y = int(
                    center_offset_y * -MOVEMENT_MULTIPLIER * COORDINATE_SCALE // self.frame_height
                )

        if target_bbox in self.stop_box:
            speed = 0.1
        else:
            speed = 1

        logger.debug(f"Movement calculated: X={self.movement_x}, Y={self.movement_y}")

    def _reset_movement(self):
        """Reset all movement commands to zero."""
        self.movement_x = 0
        self.movement_y = 0
        self.zoom_level = 0

    def _draw_debug_info(self, frame):
        """Draw debug information on frame."""
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

    def __del__(self):
        self.shared_memory.close()
        self.shared_memory.unlink()

    def run(self):
        """Main tracking loop."""
        stream = VideoStream(self.stream_path)
        
        try:
            logger.info("Starting carriage tracker...")
            while True:
                
                # wait message
                if not self.tracked:
                    self.tracked, self.descriptor, position = self.get_info()
                    self.uart.send_coordinates(*position)
                    continue
                
                # get video frame
                frame = stream.read()

                if frame is not None:
                    logger.warning("No frame received")
                    continue
                    
                processed_frame = self.preprocess_frame(frame)

                try:
                    detections_results = self.model.track(
                        processed_frame,
                        conf=DETECTOR_CONF,
                        iou=DETECTOR_IOU,
                        device="cuda:0")

                    target_found = True

                    for result in detections_results[0]:    
                        if result.boxes.cls != DRONE_CLASS_ID:
                            continue
                        
                        if self.current_target_id is None:
                            if self.update_target_id(processed_frame, result):
                                target_found = True
                        else:
                            self.calculate_movement(result)
                            target_found = True
                            break

                    if not target_found:
                        self._reset_movement()
                        logger.debug("No target found, resetting movement")
                    
                    if self.movement_x != 0 or self.movement_y != 0:
                        self.uart.send_coordinates(self.movement_x, self.movement_y)
                    
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
        except Exception as e:
            logger.error(f"Unexpected error in tracking loop: {e}")
        finally:
            logger.info("Cleaning up...")
            stream.stop()
            cv2.destroyAllWindows()

def main():
    """Main function for standalone execution."""

    tracker = Tracker(
        stream_path=conn.AIMING,
        model_path=conn.MODEL_PATH
    )
    tracker.start()

if __name__ == "__main__":
    main()