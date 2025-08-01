import time
import logging
from multiprocessing import Process, shared_memory

from ultralytics import YOLO
import json

from src.stream import VideoStream
from src.model_utils import descriptor
import cfg.connactions as config

DRONE_CLASS_ID = 1
ANGLE_OF_VIEW = 360
CAMERA_ANGLES = 112

class Overview(Process):
    """
    Multi-camera overview system for drone detection and tracking.

    Manages multiple RTSP camera streams, performs YOLO-based object detection,
    and identifies the nearest drone object across all camera feeds. Calculates
    positional information and generates feature descriptors for tracking.
    """
    def __init__(self, login, password, cameras_ips, cameras_ports, model_path):
        super().__init__()
        self.cameras_ips = cameras_ips
        self.cameras_ports = cameras_ports
        self.num_cameras = len(cameras_ips)
        self.sector_view = ANGLE_OF_VIEW / self.num_cameras

        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        self.streams_path = [template.format(login, password, ip, port) for ip, port in zip(self.cameras_ips, self.cameras_ports)]

        self.model = YOLO(model_path, task="detect")

        self.shared_memory = None

    def get_nearest_object(self, frames, detection_results, class_id:int=DRONE_CLASS_ID) -> dict:
        """Function getting dict with information about nearest object

        Args:
            frames (list[cv2.Mat]) : frames from cameras
            detection_results : detection results like a Yolo results
            class_id (int, optional): number of class, for find nearest object. Defaults to DRONE_CLASS_ID.
        
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
    
    def send_object_info(self, object_info: dict) -> bool:
        
        if object_info["object"] is None:
            return False
        
        try:
            # formation message
            object_descriptor = object_info["descriptor"].tolist()
            x_position = int(object_info["center"][0] + self.sector_view * object_info["camera"])
            y_position = 0
            print(object_descriptor, x_position, y_position)
            message = {
                "object_descriptor": object_descriptor, 
                "x_position": x_position,
                "y_position": y_position
            }

            json_message = json.dumps(message)

            # Create shared memory block
            if self.shared_memory is None:
                self.shared_memory = shared_memory.SharedMemory(
                    name="object_data",
                    create=True, 
                    size=len(json_message)
                )

            self.shared_memory.buf[:len(json_message)] = json_message.encode('utf-8')

        except Exception as err:
            print("Send object error:", err)
    

    def run(self):
        # run Videostream threads
        streams = [VideoStream(path) for path in self.streams_path]

        try:
            while True:
                start = time.time()

                # get video frames
                frames = []
                for stream in streams:
                    frame = stream.read()
                    if frame is not None:
                        frames.append(frame)
                
                # get bboxes
                if len(frames) == self.num_cameras:
                    detection_results = self.model.predict(frames,  stream=True)
                else:
                    continue

                # get nearest object
                nearest_object = self.get_nearest_object(frames, detection_results)

                # send message to one camera process 
                if nearest_object["object"] is not None:
                    self.send_object_info(nearest_object)

                print("Total time:", time.time() - start)

        finally:
            for stream in streams:
                stream.stop()

def main():
    """Main function for running process
    """

    login = config.OVERVIEW["login"]
    password = config.OVERVIEW["password"]
    cameras_ips = [config.OVERVIEW[camera]["ip"] for camera in config.OVERVIEW if camera.startswith("camera")]
    cameras_ports = [config.OVERVIEW[camera]["port"] for camera in config.OVERVIEW if camera.startswith("camera")]
    
    overview = Overview(
        login=login, 
        password=password, 
        cameras_ips=cameras_ips, 
        cameras_ports=cameras_ports, 
        model_path=config.MODEL_PATH)
    overview.start()

if __name__ == "__main__":
    main()