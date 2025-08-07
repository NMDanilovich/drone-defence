from pathlib import Path

from ultralytics import YOLO

def export_yolo(path, batch=1):
    
    model = YOLO(path)
    model.export(
        format='engine',
        int8=True,
        data="path/to/dataset", # для калибровки в int8 используется неразмеченный датасет в размере 200 изображений
        imgsz=1024, # указывается такое же, как и для обучения. иначе сильно падает Recall 
        batch=batch # 4 для обзорных камер
        )

if __name__ == "__main__":
    model_path = "path/to/model/11n_512.pt"
    print(model_path)
    export_yolo(model_path)

