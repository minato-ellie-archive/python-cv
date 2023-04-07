import uuid
import warnings
from typing import Literal, Optional, Tuple, Union

import cv2  # type: ignore
import numpy as np

from pythoncv.io.base import BaseVideoWriter
from pythoncv.types.display import WINDOW_FLAGS_DICT, WindowFlags


class VideoWindow(BaseVideoWriter):

    def __init__(
        self,
        name: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
        type: Optional[WindowFlags] = None,
    ):
        super(VideoWindow, self).__init__()
        self.name = name or f"pythoncv-display-{uuid.uuid4()}"
        self._size = size
        self._type = type
        self._is_open = False

    def open(self):
        if self._is_open:
            warnings.warn("VideoWindow is already open")
            return
        cv2.namedWindow(self.name, WINDOW_FLAGS_DICT[self._type])
        cv2.resizeWindow(self.name, self._size[0], self._size[1])
        self._is_open = True

    def close(self):
        if not self._is_open:
            warnings.warn("VideoWindow is not open")
            return
        cv2.destroyWindow(self.name)
        self._is_open = False

    @property
    def size(self) -> Tuple[int, int]:
        if not self._is_open:
            raise RuntimeError("VideoWindow is not open")
        return cv2.getWindowImageRect(self.name)[2:]

    # TODO: Add Position property

    @size.setter
    def size(self, value: Tuple[int, int]):
        cv2.resizeWindow(self.name, value[0], value[1])

    @property
    def auto_size(self) -> bool:
        if not self._is_open:
            raise RuntimeError("VideoWindow is not open")
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE) == 1

    @property
    def fullscreen(self) -> bool:
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN) == 1

    @fullscreen.setter
    def fullscreen(self, value: bool):
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, int(value))

    @property
    def aspect_ratio(self) -> float:
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_ASPECT_RATIO)

    @aspect_ratio.setter
    def aspect_ratio(self, value: Union[float, Literal["freeratio", "keepratio"]]):
        if isinstance(value, str):
            if value == "freeratio":
                value = cv2.WINDOW_FREERATIO
            elif value == "keepratio":
                value = cv2.WINDOW_KEEPRATIO
            else:
                raise ValueError(f"Invalid aspect ratio value: {value}")
        cv2.setWindowProperty(self.name, cv2.WND_PROP_ASPECT_RATIO, value)

    @property
    def opengl(self) -> bool:
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_OPENGL) == 1

    @property
    def visible(self) -> bool:
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) == 1

    @property
    def topmost(self) -> bool:
        if not self._is_open:
            raise RuntimeError("VideoWindow is not open")
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_TOPMOST) == 1

    @property
    def vsync(self) -> bool:
        if not self._is_open:
            raise RuntimeError("VideoWindow is not open")
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VSYNC) == 1
        except cv2.error:
            return False

    @vsync.setter
    def vsync(self, value: bool):
        cv2.setWindowProperty(self.name, cv2.WND_PROP_VSYNC, int(value))

    def write(self, frame: np.ndarray):
        if not self._is_open:
            raise RuntimeError("VideoWindow is not open")
        cv2.imshow(self.name, frame)
        cv2.waitKey(1)

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        if not self._is_open:
            return f"VideoWindow(name='{self.name}', size={self._size}, is_open={self._is_open})"
        return f"VideoWindow(name='{self.name}', size={self.size}, is_open={self._is_open})"

    def __str__(self):
        return self.__repr__()
