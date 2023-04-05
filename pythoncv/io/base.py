from abc import ABCMeta, abstractmethod

import numpy as np

from pythoncv.types import VideoCaptureProperties


class BaseVideo(metaclass=ABCMeta):
    """Base class for video.

    A video in pythoncv is a generator, which yields a frame(a numpy.ndarray object) each time.

    Notes:
        Image in pythoncv is a numpy.ndarray object, which has the shape of (height, width, channel).
        The channel of the image is RGB, which is different from the channel of the image in OpenCV,
        but the same as the channel of the image in PIL and Tensorflow.

    Attributes:
        path: Path to the video file.
        fps: Frames per second.
        wait_time: Time to wait between each frame.
        info: VideoCaptureProperties object, which can be used to get and set properties of the VideoCapture object.

    Methods:
        __next__: Get the next frame.
        __iter__: Get the iterator of the video.
        __len__: Get the length of the video.
    """

    @abstractmethod
    def __next__(self):
        ...

    def __iter__(self):
        return self

    def __len__(self):
        return NotImplemented

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @fps.setter
    @abstractmethod
    def fps(self, value: float):
        ...

    @property
    def wait_time(self) -> float:
        return 1 / self.fps

    @wait_time.setter
    def wait_time(self, value: float):
        self.fps = 1 / value

    @property
    @abstractmethod
    def info(self) -> VideoCaptureProperties:
        ...

    @info.setter
    @abstractmethod
    def info(self, value):
        ...


class BaseVideoWriter(metaclass=ABCMeta):
    """Base class for writer.

    Notes:
        Image in pythoncv shoule be a numpy.ndarray object, which has the shape of (height, width, channel).
        The channel of the image is RGB, which is different from the channel of the image in OpenCV,
        but the same as the channel of the image in PIL and Tensorflow.

    Args:
        path: Path to the video file.
        fps: Frames per second.
        frame_size: Size of the video frame.
        is_color: Whether the video is color or not.

    Methods:
        write: Write a frame to the video.
    """

    @abstractmethod
    def write(self, frame: np.ndarray):
        """Write a frame to the video.

        Args:
            frame: Frame to write.

        Raises:
            TypeError: If frame is not a numpy.ndarray object.
            ValueError: If the shape of frame is not (height, width, channel).
        """
        ...
