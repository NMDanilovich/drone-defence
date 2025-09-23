import time
from dataclasses import dataclass

@dataclass
class TrackObject:
    abs: tuple
    box: tuple
    tracked: bool = False
    error: tuple = (None, None)

    time: float = time.time()
    timeout = 15 # sec

    def update(self,  abs=None, box=None, error=None, tracked=None):
        self.tracked = self.tracked if tracked is None else tracked
        self.error = self.error if error is None else error
        self.abs = self.abs if abs is None else abs
        self.box = self.box if box is None else box
        self.time = time.time()

    def to_dict(self):
        return self.__dict__