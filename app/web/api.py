
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

from app.camera_controll.tracker import Tracker, FRAME
from app.camera_controll.overview import Overview
from app.camera_controll.configs import ConnactionsConfig
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
config = ConnactionsConfig()

logins = [config.data[camera]["login"] for camera in config.data if camera.startswith("OV")]
passwords = [config.data[camera]["password"] for camera in config.data if camera.startswith("OV")]
cameras_ips = [config.data[camera]["ip"] for camera in config.data if camera.startswith("OV")]
cameras_ports = [config.data[camera]["port"] for camera in config.data if camera.startswith("OV")]

overview = Overview(
    logins=logins, 
    passwords=passwords, 
    cameras_ips=cameras_ips, 
    cameras_ports=cameras_ports, 
)
overview.start()

login = config.T_CAMERA_1["login"]
password = config.T_CAMERA_1["password"]
ip = config.T_CAMERA_1["ip"]
port = config.T_CAMERA_1["port"]

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
        print(FRAME)
        if FRAME is not None:
            _, buffer = cv2.imencode('.jpg', FRAME)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(),
                    media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8085)
