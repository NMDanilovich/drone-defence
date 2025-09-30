import time

from app.camera_controll.sources.stream import VideoStream

import numpy as np

def test_exists():
    case = "./examples/stream.mp4"

    stream = VideoStream(case)

    start = time.time()

    while time.time() - start < 5:

        assert isinstance(stream.read(), np.ndarray)

def test_no_exists():
    case = "./no/path/exists"

    stream = VideoStream(case)

    assert isinstance(stream.read(), np.ndarray)

    time.sleep(1)

    assert isinstance(stream.read(), np.ndarray)