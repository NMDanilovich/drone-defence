"""
This script starts the tracker service.

It launches the `tracker.py` script in a new process.
"""
from pathlib import Path
from subprocess import Popen

library = Path(__file__).with_name("camera_controll").absolute()

tracker_service = Popen(["python3", f"{library}/tracker.py", "--core", "--debug"])

tracker_service.wait()

