
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse

app = FastAPI()
current_dir = os.path.dirname(os.path.abspath(__file__))

from app.camera_controll.tracker import Tracker
from app.camera_controll.configs import ConnactionsConfig
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
connactions = ConnactionsConfig()
login = connactions.T_CAMERA_1["login"]
password = connactions.T_CAMERA_1["password"]
ip = connactions.T_CAMERA_1["ip"]
port = connactions.T_CAMERA_1["port"]

tracker = Tracker(
    login=login,
    password=password,
    camera_ip=ip,
    camera_port=port,
)
tracker.start()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def gen_frames():
    while True:
        if tracker.frame is not None:
            _, buffer = cv2.imencode('.jpg', tracker.frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(),
                    media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
