from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar

import numpy as np

from pythoncv.core.types import VideoCaptureProperties

TFunctionOutput = TypeVar('TFunctionOutput')


class CVImage(np.ndarray):
    """ Image in pythoncv.

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
        """ Create a CVImage object from a numpy.ndarray object.

        Args:
            x:
                A numpy.ndarray object. The shape of the array should be (height, width, channel).
                The channel of the image is RGB, which is different from the channel of the image in OpenCV,
                but the same as the channel of the image in PIL and Tensorflow.
        Returns:
            A CVImage object.
        """
        return x.view(cls)

    def to_numpy(self, dtype=None, **kwargs) -> np.ndarray:
        """ Convert the CVImage object to a numpy.ndarray object.

        Args:
            dtype: Data type of the returned array.
            **kwargs: Other arguments.

        Returns:
            A numpy.ndarray object.

        """
        return self.view(np.ndarray).astype(dtype)

    def __array__(self, dtype=None, **kwargs) -> np.ndarray:
        return self.to_numpy(dtype, **kwargs)

    def then(self, fn: Callable[..., TFunctionOutput]) -> TFunctionOutput:
        return fn(self)


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
