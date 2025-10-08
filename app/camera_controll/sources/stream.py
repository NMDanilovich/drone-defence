from threading import Thread
import logging
import time

import cv2
import numpy as np

class VideoStream:
    """VideoStream class. Run thread for RTSP Stream. 
    """
    def __init__(self, stream_path=0, gst=False):
        """VideoStream class. Run thread for RTSP Stream.

        Args:
            stream_path (int, optional): Path to video or videostream. Defaults to 0.

        Raises:
            cv2.error: Could not open camera

        Examples:
        >>> path = "rtsp://login:password@ip:port/path/to/stream"
        >>> stream = VideoStream(path)
        >>> frame = stream.read()
        >>> stream.stop()
        """
        self.stream_path = stream_path
        self.frame = None

        self.collect_info(self.stream_path)

        if self.is_running:
            logging.info(f"Initializate of stream {self.stream_path}")
            
            if gst and str(self.stream_path).startswith("rtsp") :
                pipeline = self.create_rtsp_pipeline(self.stream_path)
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            elif gst and str(self.stream_path).startswith("/dev"):
                pipeline = self.create_nvargus_pipeline(self.stream_path, self.width, self.height, self.fps)
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            elif gst:
                raise ValueError("Please, specify path rtsp or /dev/* !")
            else:
                self.cap = cv2.VideoCapture(self.stream_path)

            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
        
        else:
            logging.error(f"Could not open camera {self.stream_path}")



    def collect_info(self, path):
        temp_cap = cv2.VideoCapture(self.stream_path, cv2.CAP_FFMPEG)

        if temp_cap.isOpened():
            self.width = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = temp_cap.get(cv2.CAP_PROP_FPS)
            self.is_running = True
        else:
            self.is_running = False

        temp_cap.release()

    def update(self):
        """Update frame loop in videostream"""
        while self.is_running:

            ret, frame = self.cap.read()
            if not ret:
                logging.info(f"Error: Failed to read frame from camera {self.stream_path}")
                self.is_running = False
                break
            
            self.frame = frame
            #time.sleep(0.005) # Small delay to prevent excessive CPU usage
        logging.info(f"End of stream {self.stream_path}")

    def read(self) -> np.ndarray:
        """Read actual frame
        Returns:
            np.ndarray
        """  
        return self.frame

    def stop(self):
        """Release the videostream.
        """
        if self.is_running:
            self.is_running = False
            self.thread.join()
            self.cap.release()

    @staticmethod
    def create_rtsp_pipeline(url):
        """Optimized RTSP pipeline for Jetson Orin"""
        return (
            f"rtspsrc location={url} latency=0 ! "
            "rtph265depay !"
            "queue max-size-buffers=1 ! "
            "h265parse ! "
            "nvv4l2decoder enable-max-performance=1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    @staticmethod
    def create_nvargus_pipeline(path, cap_width=1920, cap_height=1080, frame_width=1920, frame_height=1080, fps=60):
        """Optimized nvargus pipeline"""
        return (
            f"nvarguscamerasrc sensor-id={path} sensor-mode=0! "
            f"video/x-raw(memory:NVMM), width=(int){cap_width}, height=(int){cap_height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, width=(int){frame_width}, height=(int){frame_height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink max-buffers=1 drop=True"
        )