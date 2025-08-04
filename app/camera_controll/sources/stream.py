from threading import Thread
import time

import cv2

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
            self.stopped = True
            raise cv2.error(f"Error: Could not open camera {stream_path}")
            
        self.stopped = False
        self.frame = None
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Update frame loop in videostream"""
        while True:
            if self.stopped:
                break
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: Failed to read frame from camera {self.stream_path}")
                self.stopped = True
                break
            self.frame = frame
            time.sleep(0.01) # Small delay to prevent excessive CPU usage

    def read(self) -> cv2.Mat:
        """Read actual frame
        Returns:
            cv2.Mat
        """  
        return self.frame

    def stop(self):
        """Release the videostream.
        """
        self.stopped = True
        self.thread.join()
        self.cap.release()

        