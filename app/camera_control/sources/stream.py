from threading import Thread

import cv2
import numpy as np

from .logs import get_logger

logger = get_logger("Stream", terminal=False)

class VideoStream:
    """VideoStream class. Run thread for RTSP Stream. 
    """
    def __init__(self, stream_path=0, in_frame=(1080, 1920), out_frame=(576, 1024), fps=60, gst=False):
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

        logger.info(f"Initializate of stream {self.stream_path}")
        
        if gst and str(self.stream_path).startswith("rtsp") :
            pipeline = self.create_rtsp_pipeline(self.stream_path, out_frame[1], out_frame[0])
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif gst and str(self.stream_path).startswith("/dev"):
            width, height = in_frame[1], in_frame[0]
            pipeline = self.create_nvargus_pipeline(capture_width=width, capture_height=height, output_width=out_frame[1], output_height=out_frame[0], framerate=fps)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif gst:
            raise ValueError("Please, specify path rtsp or /dev/* !")
        else:
            self.cap = cv2.VideoCapture(self.stream_path, cv2.CAP_FFMPEG)

        if self.cap.isOpened():
            self.is_running = True
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
    
        else:
            self.is_running = False
            logger.error(f"Could not open camera {self.stream_path}")

    def update(self):
        """Update frame loop in videostream"""
        while self.is_running:

            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Error: Failed to read frame from camera {self.stream_path}")
                self.is_running = False
                break
            
            self.frame = frame
            #time.sleep(0.005) # Small delay to prevent excessive CPU usage
        logger.info(f"End of stream {self.stream_path}")

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
    def create_rtsp_pipeline(url, output_width:int=1024, output_height:int=576):
        """Optimized RTSP pipeline for Jetson Orin"""
        return (
            f"rtspsrc location={url} latency=0 ! "
            "rtph265depay !"
            "queue max-size-buffers=1 ! "
            "h265parse ! "
            "nvv4l2decoder enable-max-performance=1 ! "
            "nvvidconv ! "
            f"video/x-raw, width=(int){output_width}, height=(int){output_height}, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    @staticmethod
    def create_nvargus_pipeline(capture_width=2560, capture_height=1440, output_width=1920, output_height=1080, framerate=30):
        """Optimized nvargus pipeline for Jetson Orin"""
        return (
            "nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={capture_width}, height={capture_height}, framerate={framerate}/1, format=NV12 ! "
            "nvvidconv ! "
            f"video/x-raw, width=(int){output_width}, height=(int){output_height}, format=(string)BGRx !"
            "videoconvert ! video/x-raw, format=(string)BGR ! "
            "appsink max-buffers=1 drop=True"
        )
    