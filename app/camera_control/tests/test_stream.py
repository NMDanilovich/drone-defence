import time

from app.camera_control.sources.stream import VideoStream

import numpy as np
import cv2

def test_exists():
    case = "rtsp://admin:Zxcvbnm01@192.168.85.202:554/Streaming/channels/101"

    stream = VideoStream(case)
    
    time.sleep(3)
    start = time.time()
    
    while time.time() - start < 5:

        assert isinstance(stream.read(), np.ndarray)

    stream.stop()

def test_no_exists():
    """Opened false path and """
    case = "./no/path/exists"

    stream = VideoStream(case)

    time.sleep(3)
    assert stream.read() is None

    stream.stop()

def test_rtsp_pipeline():
    """Testing open and read frames from rtsp pipeline"""

    url = "rtsp://admin:Zxcvbnm01@192.168.85.202:554/Streaming/channels/101"

    cap = cv2.VideoCapture(url)

    assert cap.isOpened(), "No video from source"

    cap.release()
    
    pipeline = VideoStream.create_rtsp_pipeline(url)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    assert cap.isOpened(), f"No worked pipline: {pipeline}"

    first_ret, first_frame = cap.read()
    second_ret, second_frame = cap.read()

    assert (first_ret, second_ret) == (True, True)
    assert (first_frame - second_frame).all() == False

    cap.release()

def test_nvargus_pipeline():
    """Testing open and read frames from nvargus pipeline"""

    path = "/dev/video0"

    # cap = cv2.VideoCapture(path)

    # assert cap.isOpened(), "No video from source"
    
    # cap.release()

    pipeline = VideoStream.create_nvargus_pipeline(path, fps=60)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    assert cap.isOpened(), f"No worked pipline: {pipeline}"

    first_ret, first_frame = cap.read()
    second_ret, second_frame = cap.read()

    assert (first_ret, second_ret) == (True, True)
    assert (first_frame - second_frame).all() == False

    cap.release()

def test_gst_stream():
    """This is just the lounch streams"""

    path = "/dev/video0"
    url = "rtsp://admin:Zxcvbnm01@192.168.85.202:554/Streaming/channels/101"

    stream1 = VideoStream(path, gst=True)
    stream2 = VideoStream(url, gst=True)

    stream1.stop()
    stream2.stop()
