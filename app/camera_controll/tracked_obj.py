import time
from dataclasses import dataclass

@dataclass
class TrackObject:
    rel: tuple
    abs: tuple
    box: tuple

    time: float = time.time()
    timeout = 15 # sec

    def update(self, rel=None, abs=None, box=None):
        self.rel = self.rel if rel is None else rel
        self.abs = self.abs if abs is None else abs
        self.box = self.box if box is None else box
        self.time = time.time()

    def to_dict(self):
        return self.__dict__