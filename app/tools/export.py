from pathlib import Path

from ultralytics import YOLO

MODELS_PATH = Path(__file__).parent.parent / "models"

def export_yolo(path):
    
    model = YOLO(path)
    model.export(
        format='engine',
        int8=True,
        data="path/to/dataset", # для калибровки в int8 используется неразмеченный датасет в размере 200 изображений
        imgsz=640, # указывается такое же, как и для обучения. иначе сильно падает Recall 
        batch=1 # 4 для обзорных камер
        )

if __name__ == "__main__":
    model_path = MODELS_PATH / "11n_512.pt"
    print(model_path)
    export_yolo(model_path)

