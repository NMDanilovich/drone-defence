import time
import argparse

import requests
from requests.auth import HTTPDigestAuth


class HickTemplate:
    """XML Templates for HickVision PTZ Camera.
    """

    # --- XML Templates ---
    DEFAULT = None
    REQ_XML = {
        "Content-Type": "application/xml",
    }
    
    # --- Payloads ---
    MOMENTARY = """ 
    <PTZData>
        <pan>{}</pan>
        <tilt>{}</tilt>
        <zoom>{}</zoom>
        <Momentary>
            <duration>{}</duration>
        </Momentary>
    </PTZData>
    """

    ABSOLUTE = """
    <PTZData>
        <AbsoluteHigh>
            <azimuth>{}</azimuth>
            <elevation>{}</elevation>
            <absoluteZoom>{}</absoluteZoom>
        </AbsoluteHigh>
    </PTZData>
    """

    CONTINOUOS = """
    <PTZData>
        <pan>{}</pan>
        <tilt>{}</tilt>
        <zoom>{}</zoom>
    </PTZData>"
    """


class HickClient:
    """HickVision client for connection PTZ camera. Use the ISAPI protocol.
    """

    def __init__(self, ip, login, password):

        self.base_url: str = f"http://{ip}"
        self.login = login
        self.password = password

        self.verify_ssl = True

    @property
    def auth(self):
        return HTTPDigestAuth(self.login, self.password)

    def request(self, method, path, headers=HickTemplate.DEFAULT, data=None, json=None):

        url = self.base_url + path

        res = requests.request(
            method=method,
            headers=headers,
            auth=self.auth,
            url=url,
            data=data,
            json=json,
            verify=self.verify_ssl
        )

        return res

    def absolute_move(self, azimuth:int=0, elevation:int=0, absoluteZoom:int=1):

        if not isinstance(azimuth, int) or not isinstance(elevation, int) or not isinstance(absoluteZoom, int):
            raise ValueError("azimuth and elevation should be intager.")

        method = "PUT"
        path = "/ISAPI/PTZCtrl/channels/1/Absolute"
        headers = HickTemplate.REQ_XML
        payload = HickTemplate.ABSOLUTE.format(azimuth, elevation, absoluteZoom)
        
        res = self.request(method, path, headers, payload)

        return res

    def relative_move(self, pan:int=0, tilt:int=0, zoom:int=0, duration:int=500):

        if not isinstance(pan, int) or not isinstance(tilt, int) or not isinstance(zoom, int):
            raise ValueError("Pan, Tilt and Zoom should be intager.")
        
        method = "PUT"
        path = "/ISAPI/PTZCtrl/channels/1/Momentary"
        headers = HickTemplate.REQ_XML
        payload = HickTemplate.MOMENTARY.format(pan, tilt, zoom, duration)
        start = time.time()
        res = self.request(method, path, headers, payload)
        print("Request time:", time.time() - start)
        return res


    def continuous_move(self, pan:int=0, tilt:int=0, zoom:int=0):

        if not isinstance(pan, int) or not isinstance(tilt, int) or not isinstance(zoom, int):
            raise ValueError("Pan, Tilt and Zoom should be intager.")

        method = "PUT"
        path = "/ISAPI/PTZCtrl/channels/1/Continuous"
        headers = HickTemplate.REQ_XML
        payload = HickTemplate.CONTINOUOS.format(pan, tilt, zoom)

        res = self.request(method, path, headers, payload)

        return res
    
    def goto(self, preset: int):

        if not isinstance(preset, int) :
            raise ValueError("Preset and Tilt should be intager.")

        method = "PUT"
        path = f"/ISAPI/PTZCtrl/channels/1/presets/{preset}/goto"
        headers = HickTemplate.REQ_XML

        res = self.request(method, path,headers)

        return res

    def get_capabilities(self):

        method = "GET"
        path = "/ISAPI/PTZCtrl/channels/1/capabilities"
        headers = HickTemplate.DEFAULT

        start = time.time()
        res = self.request(method, path, headers)
        print("Request time:", time.time() - start)

        return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pan', type=int)
    parser.add_argument('-t', '--tilt', type=int)
    parser.add_argument('-z', '--zoom', type=int)
    parser.add_argument('-d', '--duration', type=int, required=False)

    args = parser.parse_args()

    ip = "192.168.85.201"
    login = "admin"
    password = "Zxcvbnm01"
    client = HickClient(ip, login, password)
    # client.get_capabilities()

    client.relative_move(args.pan, args.tilt, args.zoom, args.duration)
    client.continuous_move(args.pan, args.tilt, args.zoom)
