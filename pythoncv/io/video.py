import os
from abc import ABCMeta, abstractmethod
from typing import Union, Optional, Tuple

import cv2
import numpy as np

from pythoncv.types import (
    VideoCaptureProperties,
    VideoWriterProperties,
    CaptureBackends,
    CAPTURE_BACKEND_DICT,
    FourCC,
)
from pythoncv.io.base import BaseVideo, BaseVideoWriter


def _generate_capture_info_wrapper(cap: cv2.VideoCapture):
    """Captures the properties of the VideoCapture object.

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
            if cap.set(getattr(cv2, properties.__fields__[key].alias), getattr(properties, key)):
                return
            else:
                raise RuntimeError(f'Failed to set {key} to {value}')
        else:
            raise AttributeError(f'{properties.__class__.__name__} has no attribute {key}')

    def __repr__(self: VideoCaptureProperties):
        return str(f"VideoCaptureProperties("
                   f"fps: {self.fps}, width: {self.frame_width}, height: {self.frame_height}, "
                   f"frame_count: {self.frame_count})")

    return type(
        'VideoCaptureProperties', (object,), {
            '__doc__': VideoCaptureProperties.__doc__,
            '__getattribute__': __getattribute__,
            '__setattr__': __setattr__,
            '__repr__': __repr__,
        })()


class Video(BaseVideo):
    """Pythonic API for video.

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
        info:
            VideoCaptureProperties object, which can be used to get and set properties of the VideoCapture object.
            Some properties may not be supported by the backend.
            This time, no error will be reported, but the value will not be changed.

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

        self._info = _generate_capture_info_wrapper(self._cap)

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @fps.setter
    def fps(self, value: float):
        if self._cap.set(cv2.CAP_PROP_FPS, value):
            self._wait_time = 1 / value if value > 0 else 0
        else:
            raise RuntimeError(f'Failed to set fps to {value}')

    @property
    def info(self) -> VideoCaptureProperties:
        return self._info

    def __next__(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise StopIteration

    def __len__(self) -> Optional[int]:
        length = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length > 0:
            return length
        else:
            raise ValueError("The video has unlimited length or the length is undefined.")

    def __del__(self):
        self._cap.release()


def read_video_from_device(
    device: int,
    backend: CaptureBackends = "auto",
) -> Video:
    """Read video from a device.

    Args:
        device: Device number. Most times, your camera is 0.
        backend: Backend to use for capturing video.

    Notes:
        Some parameters of info(e.g. width, height) can not be set when the video is read by some backends.

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
        try:
            device = int(device)
        except ValueError:
            raise TypeError(f"device must be an int or str, not {type(device)}")
    elif not isinstance(device, int):
        raise TypeError(f"device must be an int or str, not {type(device)}")

    if device < 0:
        raise ValueError(f"device must be a positive integer, not {device}")

    return Video(device, backend=backend)


def read_video_from_file(
    path: Union[str, os.PathLike],
    backend: CaptureBackends = "auto",
) -> Video:
    """Read video from a file.

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
        raise FileNotFoundError(f"file not found: {os.path.abspath(path)}")

    return Video(path, backend=backend)


def read_video_from_url(
    url: str,
    backend: CaptureBackends = "auto",
) -> Video:
    """Read video from a url.

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


def _generate_writer_info_wrapper(writer: cv2.VideoWriter) -> VideoWriterProperties:
    properties = VideoWriterProperties()

    def __getattribute__(self, item):
        if item in properties.__fields__.keys():
            setattr(properties, item, writer.get(getattr(cv2, properties.__fields__[item].alias)))
            return getattr(properties, item)
        else:
            raise AttributeError(f'{properties.__class__.__name__} has no attribute {item}')

    def __setattr__(self, key, value):
        if key in properties.__fields__.keys():
            setattr(properties, key, value)
            if writer.set(getattr(cv2, properties.__fields__[key].alias), getattr(properties, key)):
                return
            else:
                raise RuntimeError(f'Failed to set {key} to {value}')
        else:
            raise AttributeError(f'{properties.__class__.__name__} has no attribute {key}')

    def __repr__(self: VideoWriterProperties):
        return str(f"VideoWriterProperties("
                   f"quality={self.quality}, n_frames={self.n_frames}, frame_bytes={self.frame_bytes})")

    return type(
        'VideoWriterProperties', (object,), {
            '__doc__': VideoWriterProperties.__doc__,
            '__getattribute__': __getattribute__,
            '__setattr__': __setattr__,
            '__repr__': __repr__,
        })()


class VideoWriter(BaseVideoWriter):
    """Pythonic API for video writer.

    Notes:
        OpenCV VideoWriter will be released automatically when the VideoWriter object is deleted,
        by using the GC mechanism.

    Args:
        path: Path to the video file.
        fps: Frames per second.
        frame_size: Size of the video frame. The shape of the frame should be (height, width).
        is_color: Whether the video is color or not.

    Methods:
        write: Write a frame to the video.

    Examples:
        >>> writer = VideoWriter("video.mp4", fps=30, frame_size=(640, 480), is_color=True)
        >>> for _ in range(30):
        >>>     writer.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    See Also:
        https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: FourCC = "mp4v",
        is_color: bool = True,
    ):
        self._writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            frame_size[::-1],
            is_color,
        )
        assert self._writer.isOpened(), AttributeError(f"failed to open video writer {path}")

        self._path = str(path)
        self._fps = fps
        self._frame_size = frame_size
        self._fourcc = fourcc
        self._is_color = is_color

        self._info = _generate_writer_info_wrapper(self._writer)

    @property
    def path(self) -> str:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_size

    @property
    def fourcc(self) -> FourCC:
        return self._fourcc

    @property
    def info(self) -> VideoWriterProperties:
        return self._info

    def write(self, frame: np.ndarray):
        assert frame.shape[:2] == self.frame_size, ValueError(
            f"frame size must be {self.frame_size}, not {frame.shape[:2]}")
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def __del__(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._writer.release()
