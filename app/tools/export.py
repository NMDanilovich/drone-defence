from pathlib import Path
from tensorrt import tensorrt

tensorrt

from ultralytics import YOLO

def export_yolo(path, batch=1):
    
    model = YOLO(path)
    model.export(
        format='engine',
        int8=True,
        data="/home/jetson/drone-defence/app/tools/calib/images/data.yaml", # для калибровки в int8 используется неразмеченный датасет в размере 200 изображений
        imgsz=(576, 1024), # указывается такое же, как и для обучения. иначе сильно падает Recall 
        batch=1, # 4 для обзорных камер
        dynamic=False,
        amp=False,
        )

if __name__ == "__main__":
    model_path = "/home/jetson/drone-defence/models/11s_1024.pt"
    print(model_path)
    export_yolo(model_path)

