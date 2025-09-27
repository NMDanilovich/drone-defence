import argparse
from pathlib import Path
import datetime
from threading import Thread

import cv2
import zmq
import numpy as np

from sources import VideoStream
from configs import ConnactionsConfig

directory = Path(__file__).parent
RESULTS = directory / "results"
RESULTS.mkdir(exist_ok=True)

class Visualization(Thread):
    def __init__(self):
        super().__init__()
        self.context = zmq.Context.instance()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.CONFLATE, 1)
        self.subscriber.connect(f"tcp://127.0.0.1:8000")
        self.subscriber.subscribe("")

        connactions = ConnactionsConfig()
        self.video_stream = VideoStream(connactions.CAMERA_4["path"])

        self.frame = None
        self.width = 1920
        self.hieght = 1080

        self._save_dir = Path("results/")
        self._save_dir.mkdir(exist_ok=True)

        name = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        self.save_path = RESULTS / f"{name}.avi"

        self.running = True

    def drow_info(self, frame, info=None):
        
        cv2.line(
            frame, 
            (self.width // 2, self.hieght // 2 - 100), 
            (self.width // 2, self.hieght // 2 + 100),
            (0, 0, 0),
            2,
            cv2.LINE_4
            
        )
        cv2.line(
            frame, 
            (self.width // 2 + 100, self.hieght // 2), 
            (self.width // 2 - 100, self.hieght // 2 ),
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
                x, y, w, h = [int(c) for c in bbox]
                pt1 = (x - w // 2, y - h // 2)
                pt2 = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame, "target", pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(frame, str((x, y)), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            else:
                color = (0, 0, 255)
            
            cv2.putText(frame, f"Track: {tracked}", (25, 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(frame, f"Error: {error}", (25, 60), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(frame, f"Detection time: {datetime.datetime.fromtimestamp(det_time)}", (25, 90), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.putText(frame, f"Current time: {datetime.datetime.now()}", (25, 120), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        return frame

    def stop(self):
        self.running = False

    def run(self, show=False, write=False):
        if write:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.save_path, fourcc, 30.0, (self.width, self.hieght), True)
            
        try:
            while self.running:
                temp_frame = self.video_stream.read()


                if temp_frame is None:
                    continue

                try:
                    data = self.subscriber.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    data = None

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
