from typing import Iterable

class BBox(object):
    def __init__(self, x:int, y:int, w:int, h:int, integer=True):
        if integer:
            self._xywh = int(x), int(y), int(w), int(h)
            self._xyxy = x - w // 2,    y - h // 2,    x + w // 2,    y + h // 2
        else:
            self._xywh = float(x), float(y), float(w), float(h)
            self._xyxy = x - w / 2,    y - h / 2,    x + w / 2,    y + h / 2


    def __getitem__(self, index):
        return self._xywh[index]

    def __contains__(self, temp):
        
        if isinstance(temp, BBox):
            x1, y1, x2, y2 = temp.xyxy
            pt1_condition = (self._xyxy[0] <= x1 <= self._xyxy[2]) and (self._xyxy[1] <= y1 <= self._xyxy[3])
            pt2_condition = (self._xyxy[0] <= x2 <= self._xyxy[2]) and (self._xyxy[1] <= y2 <= self._xyxy[3])

            return pt1_condition and pt2_condition

        elif isinstance(temp, Iterable) and 2 <= len(temp) <= 4:
            ptx, pty, *other = temp
            x_condition = self._xyxy[0] <= ptx <= self._xyxy[2]
            y_condition = self._xyxy[1] <= pty <= self._xyxy[3]

            return x_condition and y_condition
        
        else:
            raise ValueError("Point should by the two coordinates (X and Y).")
        
    def __repr__(self):
        return f"BBox(x={self._xywh[0]} y={self._xywh[1]} w={self._xywh[2]} h={self._xywh[3]})"

    @property
    def xyxy(self):
        return self._xyxy
    
    @property
    def xywh(self):
        return self._xywh
