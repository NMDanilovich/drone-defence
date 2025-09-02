from threading import Thread
import logging
import time

import cv2
import numpy as np

class VideoStream:
    """VideoStream class. Run thread for RTSP Stream. 
    """
    def __init__(self, stream_path=0):
        """VideoStream class. Run thread for RTSP Stream.

        Args:
            stream_path (int, optional): Path to video or videostream. Defaults to 0.

        Raises:
            cv2.error: Could not open camera

        Examples:
        >>> path = "rtsp://login:password:ip:port/path/to/stream"
        >>> stream = VideoStream(path)
        >>> frame = stream.read()
        >>> stream.stop()
        """
        self.stream_path = stream_path
        self.cap = cv2.VideoCapture(stream_path)
        if not self.cap.isOpened():
            self.is_running = False
            logging.error(f"Error: Could not open camera {stream_path}")
            raise cv2.error()
        else:
            self.is_running = True
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logging.info(f"Initializate of stream: w={width} h={height}")

        self.frame = None
        self.thread = Thread(target=self.update, daemon=True)

        self.thread.start()

    def update(self):
        """Update frame loop in videostream"""
        while self.is_running:

            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: Failed to read frame from camera {self.stream_path}")
                self.is_running = False
                break
            
            self.frame = frame
            time.sleep(0.005) # Small delay to prevent excessive CPU usage

    def read(self) -> np.ndarray:
        """Read actual frame
        Returns:
            np.ndarray
        """  
        return self.frame

    def stop(self):
        """Release the videostream.
        """
        self.is_running = False
        self.thread.join()
        self.cap.release()

        
