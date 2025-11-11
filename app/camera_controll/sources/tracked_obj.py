import time
from dataclasses import dataclass

@dataclass
class TrackObject:
    camera:int
    abs: tuple
    box: tuple
    id: int = None
    error: tuple = (None, None)
    tracked:bool = False
    time: float = time.time()
    timeout = 15 # sec

    def update(self, *args, **kwargs):

        for arg in kwargs:
            if arg in self.__dict__.keys(): 
                setattr(self, arg, kwargs[arg])
            else:
                raise ValueError(f"No {arg} key not in TrackingObject structure")

        self.time = time.time()

    def to_dict(self):
        return self.__dict__
