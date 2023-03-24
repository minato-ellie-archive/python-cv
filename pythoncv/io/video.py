from abc import ABCMeta, abstractmethod
from typing import Union, Literal

import numpy as np
import cv2

from pythoncv.models import VideoCaptureProperties, CaptureBackends, CAPTURE_BACKEND_DICT


def generate_info_wrapper(cap: cv2.VideoCapture):
    properties = VideoCaptureProperties()

    def __getattribute__(self, item):
        if item in properties.__fields__.keys():
            setattr(properties, item, cap.get(getattr(cv2, properties.__fields__[item].alias)))
            return getattr(properties, item)
        else:
            raise AttributeError(f'{properties.__class__.__name__} has no attribute {item}')

    def __setattr__(self, key, value):
        if key in properties.__fields__.keys():
            setattr(properties, key, value)
            cap.set(getattr(cv2, properties.__fields__[key].alias), getattr(properties, key))
        else:
            raise AttributeError(f'{properties.__class__.__name__} has no attribute {key}')

    def __repr__(self: VideoCaptureProperties):
        return str(f"VideoCaptureProperties("
                   f"fps: {self.fps}, width: {self.frame_width}, height: {self.frame_height}, "
                   f"frame_count: {self.frame_count})")

    return type('VideoCaptureProperties', (object, ), {
        '__doc__': VideoCaptureProperties.__doc__,
        '__getattribute__': __getattribute__,
        '__setattr__': __setattr__,
        '__repr__': __repr__,
    })()


class BaseVideo(metaclass=ABCMeta):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @property
    def fps(self) -> float:
        return 1 / self.wait_time

    @fps.setter
    def fps(self, value: float):
        self.wait_time = 1 / value

    @property
    @abstractmethod
    def wait_time(self) -> float:
        ...

    @wait_time.setter
    @abstractmethod
    def wait_time(self, value: float):
        ...

    @property
    @abstractmethod
    def info(self) -> VideoCaptureProperties:
        ...

    @info.setter
    @abstractmethod
    def info(self, value):
        ...


class Video(BaseVideo):
    def __init__(
        self,
        path: Union[str, int],
        backend: CaptureBackends = "auto",
    ):
        self._cap = cv2.VideoCapture(path, CAPTURE_BACKEND_DICT[backend])
        self.path = path

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._wait_time = 1 / fps if fps > 0 else 0
        self._info = generate_info_wrapper(self._cap)

    @property
    def wait_time(self) -> float:
        return self._wait_time

    @wait_time.setter
    def wait_time(self, value: float):
        self._wait_time = value

    @property
    def info(self) -> VideoCaptureProperties:
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    def __iter__(self):
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                yield frame
            else:
                break

    def __len__(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        self._cap.release()
