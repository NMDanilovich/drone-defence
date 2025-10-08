from pathlib import Path
import time

import pandas as pd
from ultralytics import YOLO

from app.camera_controll.sources.stream import VideoStream
from app.camera_controll.configs import ConnactionsConfig, SystemConfig

def get_info(detection_results, drone_cls:int):
    
    for camera_results in detection_results:
        if camera_results is None:
            continue

        for result in camera_results:    
            if result.boxes.cls == drone_cls:
                return True
    
    else:
        return False
    
def test_model_in_static():
    sys_conf = SystemConfig("./examples/test_system.conf")
    iou = 0.9
    
    conf = 1
    time_step = 3
    model = YOLO(sys_conf.MODEL["path"], task="detect")

    i = 0
    data = pd.DataFrame(columns=("confidence", "detections", "num_frames", "score"))
    
    try:
        while conf > 0:
            stream = VideoStream()
            start = time.time()
            num_pred = 0
            all_frames = 0

            while time.time() - start < time_step:
                all_frames += 1
                frame = stream.read()

                if frame is None:
                    continue

                res = model.predict(
                    frame,
                    
                    imgsz=(576, 1024),
                    conf=conf,
                    iou=iou,
                    verbose=True
                )

                if get_info(res):
                    num_pred += 1

            score = round(num_pred/all_frames, 3) 

            data.loc[i] = (round(conf, 3), num_pred, all_frames, score)
            conf -= 0.05
            i += 1
    
    finally:
        stream.stop()

        res = Path(__file__).joinpath("test_model_results")
        res.mkdir(exist_ok=True)

        data.to_csv("test_results.csv")

def test_model_speed():
    sys_conf = SystemConfig("./examples/test_system.conf")
    model = YOLO(sys_conf.MODEL["path"], task="detect")

    
