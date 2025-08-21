import time
import logging
from multiprocessing import Process

from ultralytics import YOLO
import zmq
import json

from sources import VideoStream
from sources import descriptor
from sources import coord_to_steps, coord_to_angle
from configs import ConnactionsConfig, OverviewConfig, CalibrationConfig

BUFF_SIZE = 15_000

class Overview(Process):
    """
    Manages a multi-camera overview system for detecting and tracking drones.

    This class operates as a separate process to handle multiple RTSP camera streams concurrently.
    It uses a YOLO model for object detection on each stream, identifies the nearest drone based on object area,
    and calculates its position. Feature descriptors are generated for tracking purposes.
    The information about the nearest detected object is then shared with other processes
    via shared memory.

    Args:
        logins (list[str]): A list of login usernames for the RTSP cameras.
        passwords (list[str]): A list of passwords for the RTSP cameras.
        cameras_ips (list[str]): A list of IP addresses for the RTSP cameras.
        cameras_ports (list[int]): A list of ports for the RTSP cameras.
        config_path (str, optional): Path to the overview configuration file. Defaults to None.
        calibr_path (str, optional): Path to the calibration configuration file. Defaults to None.

    Attributes:
        ov_config (OverviewConfig): Configuration for the overview system.
        calibr_config (CalibrationConfig): Configuration for camera calibration.
        num_cameras (int): The number of cameras being managed.
        streams_path (list[str]): A list of RTSP stream URLs.
        model (YOLO): The YOLO object detection model.
        shared_memory (shared_memory.SharedMemory): The shared memory block for inter-process communication.
        running (bool): A flag to control the main loop of the process.
    """
    def __init__(self, logins, passwords, cameras_ips, cameras_ports, config_path=None, calibr_path=None):
        """Manages a multi-camera overview system for detecting and tracking drones.

        Args:
            logins (list[str]): A list of login usernames for the RTSP cameras.
            passwords (list[str]): A list of passwords for the RTSP cameras.
            cameras_ips (list[str]): A list of IP addresses for the RTSP cameras.
            cameras_ports (list[int]): A list of ports for the RTSP cameras.
            config_path (str, optional): Path to the overview configuration file. Defaults to None.
            calibr_path (str, optional): Path to the calibration configuration file. Defaults to None.
        """
        super().__init__()
        self.ov_config = OverviewConfig(config_path)
        self.calibr_config = CalibrationConfig(calibr_path)

        self.num_cameras = len(cameras_ips)

        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        self.streams_path = []
        self.connactions_info = zip(logins, passwords, cameras_ips, cameras_ports)

        for login, password, ip, port in self.connactions_info:
            self.streams_path.append(template.format(login, password, ip, port))

        self.model = YOLO(self.ov_config.MODEL_PATH, task="detect")
        self.timeout = self.ov_config.TIMEOUT

        self.context: zmq.Context = None
        self.socket: zmq.Socket = None

        self.running = True

    def _init_connaction(self):
        """Initialization socket for processes connaction.
        """

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.setsockopt(zmq.RCVHWM, 10)
        self.socket.bind(f"tcp://127.0.0.1:8000")
        logging.info("Initalization Publisher socket")

    def get_nearest_object(self, frames, detection_results, class_id:int) -> dict:
        """
        Identifies the nearest detected object of a specific class across all camera frames.

        The nearest object is determined by the largest bounding box area.

        Args:
            frames (list): A list of video frames (e.g., from OpenCV) from all cameras.
            detection_results (list): A list of YOLO detection results for each frame.
            class_id (int): The class ID to search for (e.g., the ID for drones).

        Returns:
            dict: A dictionary containing information about the nearest object found.
                  The dictionary has the following structure:
                  {
                      "camera": int or None,      # Index of the camera that saw the nearest object.
                      "object": int or None,      # Index of the object in the camera's detection results.
                      "center": tuple or None,    # (x, y) coordinates of the object's center.
                      "descriptor": tensor or None # Feature descriptor for the object.
                  }
        """

        nearest_object = {
            "camera": None,
            "object": None,
            "center": None,
            "descriptor": None
        }

        max_area = 0
        for camera_index, camera_results in enumerate(detection_results):
            if camera_results is None:
                continue

            for object_index, object in enumerate(camera_results):
                if object.boxes.cls == class_id:

                    *center, w, h = object.boxes.xywh[0]
                    # x, y = center
                    object_area = w * h

                    if object_area > max_area:
                        max_area = object_area

                        nearest_object["camera"] = camera_index
                        nearest_object["object"] = object_index
                        nearest_object["center"] = center

                        x1, y1, x2, y2 = map(int, object.boxes.xyxy[0])
                        object_region = frames[camera_index][y1:y2, x1:x2]

                        if object_region.size == 0:
                            print("Very small object")
                            continue
                        
                        nearest_object["descriptor"] = descriptor(object_region)

        return nearest_object

    def cleanup(self):
        """Cleans up shared memory resources to prevent memory leaks."""
        try:
            self.context.destroy()
        except Exception as e:
            logging.error(f"Error during sender cleanup: {e}")

        finally:
            self.context = None
    
    def send_object_info(self, object_info: dict) -> None:
        """
        Serializes and sends information about the detected object via shared memory.

        This method takes the object information, calculates the target position in steps and angles
        based on calibration data, formats it into a JSON message, and writes it to the
        shared memory buffer for other processes to read.

        Args:
            object_info (dict): A dictionary containing details about the nearest object,
                                as returned by `get_nearest_object`.

        Returns:
            None
        """
        
        if object_info["object"] is None:
            return None
        
        try:
            
            # formation message
            object_descriptor = object_info["descriptor"].tolist()
            x, y = object_info["center"]

            x_calib = self.calibr_config.ALL[object_info["camera"]]
            y_calib = self.ov_config.HORIZONT

            width = self.ov_config.WIDTH_FRAME
            hor = self.ov_config.HORIZ_ANGLE
            height = self.ov_config.HEIGHT_FRAME
            vert = self.ov_config.VERTIC_ANGLE

            x_position = int(x_calib + coord_to_angle(x, width, hor)) # steps
            y_position = int(y_calib - coord_to_angle(y, height, vert)) # angles

            message = {
                "object_descriptor": object_descriptor, 
                "x_position": x_position,
                "y_position": y_position
            }

            json_message = json.dumps(message)

            # clear tail buffer
            self.socket.send_json(json_message)

        except Exception as err:
            print("Send object error:", err)
    

    def run(self):
        logging.info("Intialization of streams.")
        streams = [VideoStream(path) for path in self.streams_path]
        self._init_connaction()

        try:
            while self.running:
                start = time.time()

                # get video frames
                frames = []
                for stream in streams:
                    frame = stream.read()
                    if frame is not None:
                        frames.append(frame)
                
                # get bboxes
                if len(frames) == self.num_cameras:
                                       
                    detection_results = []
                    for frame in frames:
                        result = self.model.predict(
                            frame, 
                            conf=self.ov_config.DETECTOR_CONF,
                            iou=self.ov_config.DETECTOR_IOU,
                            )
                        detection_results.append(result[0])
                else:
                    continue

                # get nearest object
                nearest_object = self.get_nearest_object(
                    frames, 
                    detection_results, 
                    self.ov_config.DRONE_CLASS_ID
                )

                # send message to one camera process 
                if nearest_object["object"] is not None:
                    self.send_object_info(nearest_object)

                time.sleep(self.timeout)
                print("Total time:", time.time() - start)

        finally:
            for stream in streams:
                stream.stop()

            self.cleanup()

    def stop(self):
        """Stops the process."""
        self.running = False

def main():
    """Initializes and runs the Overview process."""

    config = ConnactionsConfig()

    logins = [config.data[camera]["login"] for camera in config.data if camera.startswith("OV")]
    passwords = [config.data[camera]["password"] for camera in config.data if camera.startswith("OV")]
    cameras_ips = [config.data[camera]["ip"] for camera in config.data if camera.startswith("OV")]
    cameras_ports = [config.data[camera]["port"] for camera in config.data if camera.startswith("OV")]

    overview = Overview(
        logins=logins, 
        passwords=passwords, 
        cameras_ips=cameras_ips, 
        cameras_ports=cameras_ports, 
    )
    
    try:
        overview.run()
        overview.join()
    except KeyboardInterrupt:
        overview.stop()
        overview.join()

if __name__ == "__main__":
    main()