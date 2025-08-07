import time
import logging
from multiprocessing import Process, shared_memory

from ultralytics import YOLO
import json

from sources import VideoStream
from sources import descriptor
from sources import coord_to_steps, coord_to_angle
from configs import ConnactionsConfig, OverviewConfig, CalibrationConfig

BUFF_SIZE = 15_000

class Overview(Process):
    """
    Multi-camera overview system for drone detection and tracking.

    Manages multiple RTSP camera streams, performs YOLO-based object detection,
    and identifies the nearest drone object across all camera feeds. Calculates
    positional information and generates feature descriptors for tracking.
    """
    def __init__(self, logins, passwords, cameras_ips, cameras_ports, config_path=None, calibr_path=None):
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

        self.shared_memory = None

        self.running = True

    def get_nearest_object(self, frames, detection_results, class_id:int) -> dict:
        """Function getting dict with information about nearest object

        Args:
            frames (list[cv2.Mat]) : frames from cameras
            detection_results : detection results like a Yolo results
            class_id (int, optional): number of class, for find nearest object.
        
        Returns:
            information (dict) : information about nearest object. 
            Has the structure: 
            ```
            {
                "camera": int | None, # number of camera
                "object": int | None, # number of object from camera results
                "center": tuple[int, int] | None # object center on frame
                "decriptor": torch.Tensor() | None # unique embedding for object 
            }
            ```
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
        """Clean up shared memory resources."""
        try:
            self.shared_memory.close()
            self.shared_memory.unlink()
        except FileNotFoundError:
            logging.info(f"Shared memory was already unlinked.")
        except Exception as e:
            logging.error(f"Error during sender cleanup: {e}")

        finally:
            self.shared_memory = None
    
    def send_object_info(self, object_info: dict) -> None:
        """Sending message using the shared memory.

        Args:
            object_info (dict): object info like 
            ```
            {
                "camera": int | None, # number of camera
                "object": int | None, # number of object from camera results
                "center": tuple[int, int] | None # object center on frame
                "decriptor": torch.Tensor() | None # unique embedding for object 
            }
            ```

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

            x_position = int(x_calib + coord_to_steps(x, width, hor)) # steps
            y_position = int(y_calib - coord_to_angle(y, height, vert)) # angles

            message = {
                "object_descriptor": object_descriptor, 
                "x_position": x_position,
                "y_position": y_position
            }

            json_message = json.dumps(message)
            json_bytes = json_message.encode('utf-8')

            # Create shared memory block
            if self.shared_memory is None:
                self.shared_memory = shared_memory.SharedMemory(
                    name="object_data",
                    create=True, 
                    size=BUFF_SIZE
                )
            
            # clear tail buffer
            self.shared_memory.buf[:] = b"\x00" * BUFF_SIZE

            # json information to buffer
            self.shared_memory.buf[:len(json_bytes)] = json_bytes

        except Exception as err:
            print("Send object error:", err)
    

    def run(self):
        logging.info("Intialization of streams.")
        streams = [VideoStream(path) for path in self.streams_path]

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
                    detection_results = self.model.predict(
                        frames, 
                        conf=self.ov_config.DETECTOR_CONF,
                        iou=self.ov_config.DETECTOR_IOU,
                        stream=True)
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

                print("Total time:", time.time() - start)

        finally:
            for stream in streams:
                stream.stop()

            self.cleanup()

    def stop(self):
        """Stoped process"""
        self.running = False

def main():
    """Main function for running process
    """

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
        overview.start()
        overview.join()
    except KeyboardInterrupt:
        overview.stop()
        overview.join()

if __name__ == "__main__":
    main()