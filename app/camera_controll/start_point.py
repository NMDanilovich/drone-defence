from overview import Overview
from tracker import Tracker
import configs.connactions as config

def main():
    login = config.OVERVIEW["login"]
    password = config.OVERVIEW["password"]
    cameras_ips = [config.OVERVIEW[camera]["ip"] for camera in config.OVERVIEW if camera.startswith("camera")]
    cameras_ports = [config.OVERVIEW[camera]["port"] for camera in config.OVERVIEW if camera.startswith("camera")]
    
    overview = Overview(
        login=login, 
        password=password, 
        cameras_ips=cameras_ips, 
        cameras_ports=cameras_ports, 
        model_path=config.MODEL_PATH)
    overview.start()

    carriage_camera = f"rtsp://{login}:{password}@192.168.85.201:554/Streaming/channels/101"
    tracker = Tracker(
        #stream_path=config.AIMING,
        stream_path=carriage_camera,
        model_path=config.MODEL_PATH,
        controller_path="/dev/THS0"
    )
    tracker.start()

    overview.join()
    tracker.join()

if __name__ == "__main__":
    main()