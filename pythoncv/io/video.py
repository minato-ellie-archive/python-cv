import os
from abc import ABCMeta, abstractmethod
from typing import Union, Optional, Tuple

import cv2
import numpy as np

from pythoncv.types import (
    VideoCaptureProperties,
    CaptureBackends,
    CAPTURE_BACKEND_DICT,
    FourCC,
)


def _generate_info_wrapper(cap: cv2.VideoCapture):
    """
    Generates a wrapper, which make a fake VideoCaptureProperties object, which can be used to get and set
    properties of the VideoCapture object.

    Args:
        cap: VideoCapture object

    Returns:
        A wrapper object, which can be used to get and set properties of the VideoCapture object.
    """
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

    return type('VideoCaptureProperties', (object,), {
        '__doc__': VideoCaptureProperties.__doc__,
        '__getattribute__': __getattribute__,
        '__setattr__': __setattr__,
        '__repr__': __repr__,
    })()


class BaseVideo(metaclass=ABCMeta):
    """
    Base class for video.

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
    """
    Pythonic API for video.

    Notes:
        Image in pythoncv is a numpy.ndarray object, which has the shape of (height, width, channel).
        The channel of the image is RGB, which is different from the channel of the image in OpenCV,
        but the same as the channel of the image in PIL and Tensorflow.

        OpenCV Capture will be released automatically when the Video object is deleted, by using the GC mechanism.

    Args:
        path: Path to the video file.
        backend: Backend to use for capturing video.
            If backend is "auto", the backend will be chosen automatically.
            If backend is "opencv", the backend will be OpenCV.
            If backend is "ffmpeg", the backend will be ffmpeg.
            If backend is "gstreamer", the backend will be gstreamer.
            other backend types can be found in `pythoncv.types.CaptureBackends``

    Attributes:
        path: Path to the video. If the video is read from a device, the path will be the device number.
        fps: Frames per second. When you set the fps, the wait_time will be changed automatically.
        wait_time: Time to wait between each frame.
        info: VideoCaptureProperties object, which can be used to get and set properties of the VideoCapture object.

    Methods:
        __next__: Get the next frame.
        __iter__: Get the iterator of the video.
        __len__:
            Get the length of the video.
            If the video is read from a device(which has unlimited length), the length will be set to None.
    """

    def __init__(
        self,
        path: Union[str, int],
        backend: CaptureBackends = "auto",
    ):
        self._cap = cv2.VideoCapture(path, CAPTURE_BACKEND_DICT[backend])
        self.path = path

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._wait_time = 1 / fps if fps > 0 else 0
        self._info = _generate_info_wrapper(self._cap)

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

    def __next__(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise StopIteration

    def __len__(self) -> Optional[int]:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        self._cap.release()


def read_video_from_device(
    device: int,
    backend: CaptureBackends = "auto",
) -> Video:
    """
    Read video from a device.

    Args:
        device: Device number. Most times, your camera is 0.
        backend: Backend to use for capturing video.

    Returns:
        A Video object.

    Raises:
        TypeError: If device is not an int or str.
        ValueError: If device is not a positive integer.

    Examples:
        >>> video = read_video_from_device(0)
        >>> for frame in video:
        >>>     print(frame.shape)
        (480, 640, 3)
        (480, 640, 3)
        ...
    """
    if isinstance(device, str):
        device = int(device)
    elif not isinstance(device, int):
        raise TypeError(f"device must be an int or str, not {type(device)}")

    if device < 0:
        raise ValueError(f"device must be a positive integer, not {device}")

    return Video(device, backend=backend)


def read_video_from_file(
    path: Union[str, os.PathLike],
    backend: CaptureBackends = "auto",
) -> Video:
    """
    Read video from a file.

    Args:
        path: Path to the video file.
        backend: Backend to use for capturing video.

    Returns:
        A Video object.

    Raises:
        TypeError: If path is not a string.
        FileNotFoundError: If the file is not found.

    Examples:
        >>> video = read_video_from_file("video.mp4")
        >>> for frame in video:
        >>>     print(frame.shape)
        (1080, 1920, 3)
        (1080, 1920, 3)
        ...
    """
    if not isinstance(path, str):
        try:
            path = str(path)
        except Exception as e:
            raise TypeError(f"path must be a string, not {type(path)}") from e

    if not os.path.isfile(path):
        raise FileNotFoundError(f"file {path} not found")

    return Video(path, backend=backend)


def read_video_from_url(
    url: str,
    backend: CaptureBackends = "auto",
) -> Video:
    """
    Read video from a url.

    Args:
        url: Url to the video.
        backend: Backend to use for capturing video.

    Returns:
        A Video object.

    Raises:
        TypeError: If url is not a string.

    Examples:
        >>> video = read_video_from_url("https://mazwai.com/videvo_files/video/free/2018-12/small_watermarked/180607_A_124_preview.mp4") # noqa E501
        >>> for frame in video:
        >>>     print(frame.shape)
        (1080, 1920, 3)
        (1080, 1920, 3)
        ...
    """
    if not isinstance(url, str):
        raise TypeError(f"url must be a string, not {type(url)}")

    return Video(url, backend=backend)

