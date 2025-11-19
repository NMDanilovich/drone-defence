from ultralytics import YOLO
import cv2

from app.camera_control.sources import VideoStream

def test_pipeline():
        detector = YOLO("/home/jetson/drone-defence/models/yolo11s_drone_person.engine", task="detect")
        image_size = (576, 1024)

        stream = VideoStream("rtsp://admin:Zxcvbnm01@192.168.88.80:554/Streaming/channels/101", gst=True)

        try:
            while True:
                temp_frame = stream.read()

                if temp_frame is None:
                    continue

                detector.predict(temp_frame,
                                 imgsz=image_size)


        finally:
            stream.stop()

test_pipeline()