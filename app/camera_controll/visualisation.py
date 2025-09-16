from pathlib import Path

import cv2
import zmq
import numpy as np

from sources import VideoStream

class Visualization:
    def __init__(self):
        self.context = zmq.Context.instance()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://127.0.0.1:8000")
        self.subscriber.subscribe("")

        # Hardcoded for now, should be dynamic
        self.video_stream = VideoStream("rtsp://admin:Zxcvbnm01@192.168.85.206:554/Streaming/channels/101")

        self.width = 1920
        self.hieght = 1080

        self._save_dir = Path("results/")
        self._save_dir.mkdir(exist_ok=True)

        self.save_path =  self._save_dir / "output.avi"

    def drow_info(self, frame, info=None):
        
        cv2.line(
            frame, 
            (self.width // 2, self.hieght // 2 - 100), 
            (self.width // 2, self.hieght // 2 + 100),
            (0, 255, 255),
            3,
            cv2.LINE_4
            
        )
        cv2.line(
            frame, 
            (self.width // 2 + 100, self.hieght // 2), 
            (self.width // 2 - 100, self.hieght // 2 ),
            (0, 255, 255),
            3,
            cv2.LINE_4
                )
        
        tracked = False

        if info:
            bbox = info.get("box")
            tracked = info.get("tracked")
            
            if tracked:
                x, y, w, h = [int(c) for c in bbox]
                pt1 = (x - w // 2, y - h // 2)
                pt2 = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame, "target", pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Track: {tracked}", (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        return frame


    def run(self, show=False):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.save_path, fourcc, 20.0, (self.width, self.hieght), True)

        try:
            while True:
                frame = self.video_stream.read()


                if frame is None:
                    continue

                try:
                    data = self.subscriber.recv_json(flags=zmq.NOBLOCK)

                except zmq.Again:
                    data = None

                self.drow_info(frame, data)

                out.write(frame)
                print("Writed ...", end="\r")

                if show:
                    frame = cv2.resize(frame, (640, 420))
                    cv2.imshow("Detection and Tracking", frame)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            print(f"Complite! Save video as {self.save_path}")
            out.release()
            self.video_stream.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    vis = Visualization()
    vis.run(show=False)
