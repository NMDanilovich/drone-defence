import time

from overview import Overview
from tracker import Tracker
from configs import ConnactionsConfig

def main():
    config = ConnactionsConfig()

    logins = [config.data[camera]["login"] for camera in config.data if camera.startswith("ov")]
    passwords = [config.data[camera]["password"] for camera in config.data if camera.startswith("ov")]
    cameras_ips = [config.data[camera]["ip"] for camera in config.data if camera.startswith("ov")]
    cameras_ports = [config.data[camera]["port"] for camera in config.data if camera.startswith("ov")]
    
    overview = Overview(
        logins=logins, 
        passwords=passwords, 
        cameras_ips=cameras_ips, 
        cameras_ports=cameras_ports, 
    )
    overview.start()
    
    time.sleep(3)
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
    
    tracker.join()
    overview.join()
    

if __name__ == "__main__":
    main()