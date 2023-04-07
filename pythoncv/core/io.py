from abc import ABCMeta, abstractmethod

import numpy as np

from pythoncv.core.types import VideoCaptureProperties


class CVImage(np.ndarray):
    """ A Monad for numpy.ndarray.

    Notes:
        Image in pythoncv is a numpy.ndarray object, which has the shape of (height, width, channel).
        The channel of the image is RGB, which is different from the channel of the image in OpenCV,
        but the same as the channel of the image in PIL and Tensorflow.

    Methods:
        from_numpy: Create a CVImage object from a numpy.ndarray object.
        then: Apply a function to the CVImage object.
    """

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        return x.view(cls)

    def then(self, fn):
        return fn(self)

    def __repr__(self):
        return f"CVImage({super().__repr__()})"

    def __str__(self):
        return f"CVImage({super().__str__()})"


class CVVideo(metaclass=ABCMeta):
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
    def __next__(self) -> CVImage:
        ...  # pragma: no cover

    def __iter__(self):
        return self

    def __len__(self):
        return NotImplemented  # pragma: no cover

    @property
    @abstractmethod
    def fps(self) -> float:
        ...  # pragma: no cover

    @fps.setter
    @abstractmethod
    def fps(self, value: float):
        ...  # pragma: no cover

    @property
    def wait_time(self) -> float:
        return 1 / self.fps

    @wait_time.setter
    def wait_time(self, value: float):
        self.fps = 1 / value

    @property
    @abstractmethod
    def info(self) -> VideoCaptureProperties:
        ...  # pragma: no cover
