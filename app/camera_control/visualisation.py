import argparse
from pathlib import Path
import datetime
from threading import Thread

import cv2
import zmq
import numpy as np

from sources.logs import get_logger, LOGS_DIRECTORY
from sources import VideoStream
from configs import ConnectionsConfig

logger = get_logger("Visual_serv")

RESULTS = LOGS_DIRECTORY.parent.joinpath("results")
RESULTS.mkdir(exist_ok=True)

class Visualization(Thread):
    """
    A class to visualize the video stream with object tracking information.

    This class runs as a separate thread to continuously read frames from a video stream,
    receive tracking information, draw bounding boxes and other data on the frames,
    and save the output as a video file.
    """
    def __init__(self):
        """
        Initializes the Visualization thread.

        Sets up the ZeroMQ subscriber for receiving tracking data, initializes the video stream,
        and prepares for saving the output video.
        """
        super().__init__()
        self.context = zmq.Context.instance()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.CONFLATE, 1)
        self.subscriber.connect(f"tcp://127.0.0.1:8000")
        self.subscriber.subscribe("")

        connections = ConnectionsConfig()
        camera = None
        for name in connections.NAMES:
            if connections.data[name]["track"]:
                camera = connections.data[name]
                break

        if camera is None:
            raise IndexError("Tracked camera is not found")
        
        template = "rtsp://{}:{}@{}:{}/Streaming/channels/101"
        path = camera["path"] if camera["path"] else template.format(camera["login"], camera["password"], camera["ip"], camera["port"])
        
        self.video_stream = VideoStream(path, gst=True)

        self.frame = None
        self.width = 1920
        self.hieght = 1080

        self._save_dir = Path("results/")
        self._save_dir.mkdir(exist_ok=True)

        name = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        self.save_path = RESULTS / f"{name}.avi"

        self.running = True

    def drow_info(self, frame, info=None):
        """
        Draws tracking information on the frame.

        Args:
            frame (np.ndarray): The video frame to draw on.
            info (dict, optional): A dictionary containing tracking information.
                                  Defaults to None.

        Returns:
            np.ndarray: The frame with the information drawn on it.
        """
        
        cv2.line(
            frame, 
            (self.width // 2, self.hieght // 2 - 50), 
            (self.width // 2, self.hieght // 2 + 50),
            (0, 0, 0),
            2,
            cv2.LINE_4
            
        )
        cv2.line(
            frame, 
            (self.width // 2 + 50, self.hieght // 2), 
            (self.width // 2 - 50, self.hieght // 2 ),
            (0, 0, 0),
            2,
            cv2.LINE_4
                )
        
        tracked = False

        if info:
            bbox = info.get("box")
            tracked = info.get("tracked")
            error = info.get("error")
            det_time = info.get("time")
            
            if tracked:
                color = (0, 255, 0)
                x, y, w, h = bbox
                x = x*self.width
                w = w*self.width
                y = y*self.width
                h = h*self.width

                pt1 = (int((x - w // 2)*self.width), int((y - h // 2)*self.hieght))
                pt2 = (int((x + w // 2)*self.width), int((y + h // 2)*self.hieght))
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame, "target", pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame, str((x, y)), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            else:
                color = (0, 0, 255)
            
            cv2.putText(frame, f"Track: {tracked}", (25, 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(frame, f"Error: {error}", (25, 60), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(frame, f"Detection time: {datetime.datetime.fromtimestamp(det_time)}", (25, 90), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(frame, f"Current time: {datetime.datetime.now()}", (25, 120), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        return frame

    def stop(self):
        """Stops the visualization thread."""
        self.running = False

    def run(self, show=False, write=False):
        """
        The main loop of the visualization thread.

        Reads frames, receives tracking data, draws information on the frames,
        and saves the output video.

        Args:
            show (bool, optional): Whether to display the video feed in a window.
                                   Defaults to False.
            write (bool, optional): Whether to write the output video to a file.
                                    Defaults to False.
        """
        if write:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(self.save_path), fourcc, 30.0, (self.width, self.hieght), True)
            
        try:
            while self.running:
                temp_frame = self.video_stream.read()


                if temp_frame is None:
                    continue

                try:
                    data = self.subscriber.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    data = None

                self.frame = temp_frame
                self.frame = self.drow_info(temp_frame, data)

                if write:
                    out.write(self.frame)
                    print("Writed ...", end="\r")

                if show:
                    self.frame = cv2.resize(self.frame, (640, 420))
                    cv2.imshow("Detection and Tracking", self.frame)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            if write:
                print(f"Complite! Save video as {self.save_path}")
                out.release()

            self.video_stream.stop()
            self.context.destroy()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualisation Service")
    
    parser.add_argument("--write", action="store_true", help="Writed video into results directory")
    parser.add_argument("--show", action="store_true", help="Open window with frames from camera")
    
    args = parser.parse_args()

    vis = Visualization()
    vis.run(write=args.write, show=args.show)
