import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Any

import cv2  # type: ignore
import numpy as np

from pythoncv.types.image import (IMAGE_READ_FLAG_DICT, IMAGE_WRITE_FLAG_DICT, ImageReadFlag, ImageWriteFlag)


def _image_read_flag_wrapper(
    color_mode: ImageReadFlag,
    reduce_ratio: Literal[None, 2, 4, 8] = None,
) -> int:
    assert color_mode in IMAGE_READ_FLAG_DICT, AttributeError(f"Invalid color_mode {color_mode}")

    flag = IMAGE_READ_FLAG_DICT[color_mode]
    if reduce_ratio is not None:
        if color_mode in ['grayscale', 'color']:
            flag = IMAGE_READ_FLAG_DICT[(color_mode, reduce_ratio)]
        else:
            raise AttributeError(f"Cannot reduce image with color_mode {color_mode}")
    return flag


def read_image_from_file(
    filename: Union[str, Path],
    color_mode: ImageReadFlag = 'unchanged',
    reduce_ratio: Literal[None, 2, 4, 8] = None,
) -> np.ndarray:
    """Read image from file.

    Args:
        filename: Path to the image
        color_mode: Color mode of the image (e.g. 'grayscale', 'color', 'unchanged', default is 'unchanged')
        reduce_ratio: Reduce ratio of the image (only None, 2, 4, 8 are valid, default is None, meaning no reduction)

    Returns:
        Image as a numpy array. (H, W, C) and dtype uint8, Channel order is RGB.
        If the image is grayscale, the shape is (H, W).

    Raises:
        AttributeError: If the image cannot be read

    Examples:
        >>> from pythoncv.io import read_image_from_file
        >>> image = read_image_from_file('path/to/image.jpg')
        >>> print(image.shape)
        (H, W, C)

    See Also:
        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    """
    flag = _image_read_flag_wrapper(color_mode, reduce_ratio)

    result = cv2.imread(str(filename), flag)
    assert result is not None, AttributeError(f"Failed to read image from {os.path.abspath(filename)}")
    return result[..., ::-1]


def read_image_from_bytes(
    b: bytes,
    color_mode: ImageReadFlag = 'unchanged',
    reduce_ratio: Literal[None, 2, 4, 8] = None,
) -> np.ndarray:
    """Read image from bytes.

    Args:
        b: Bytes of the image
        color_mode: Color mode of the image
        reduce_ratio: Reduce ratio of the image

    Returns:
        Image as a numpy array. (H, W, C) and dtype uint8, Channel order is RGB.
        If the image is grayscale, the shape is (H, W).

    Raises:
        AttributeError: If the image cannot be read

    Examples:
        >>> from pythoncv.io import read_image_from_bytes
        >>> with open('path/to/image.jpg', 'rb') as f:
        >>>     image = read_image_from_bytes(f.read())

    See Also:
        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e
    """
    flag = _image_read_flag_wrapper(color_mode, reduce_ratio)
    result = cv2.imdecode(np.frombuffer(b, np.uint8), flag)
    assert result is not None, AttributeError("Failed to read image from bytes")
    return result[..., ::-1]


def read_image(
    image: Union[bytes, str, Path],
    color_mode: ImageReadFlag = 'unchanged',
    reduce_ratio: Literal[None, 2, 4, 8] = None,
) -> np.ndarray:
    """Read image from file or bytes.

    Args:
        image: Path to the image or bytes of the image
        color_mode: Color mode of the image
        reduce_ratio: Reduce ratio of the image

    Returns:
        Image as a numpy array. (H, W, C) and dtype uint8, Channel order is RGB.
        If the image is grayscale, the shape is (H, W).

    Raises:
        AttributeError: If the image cannot be read

    Examples:
        >>> from pythoncv.io import read_image
        >>> image = read_image('path/to/image.jpg')
        >>> print(image.shape)
        (H, W, C)

    See Also:
        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56

        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e
    """
    if isinstance(image, bytes):
        return read_image_from_bytes(image, color_mode, reduce_ratio)
    else:
        return read_image_from_file(image, color_mode, reduce_ratio)


def _image_write_flag_wrapper(
    type: Optional[ImageWriteFlag] = None,
    quality: Union[None, int, float] = None,
) -> Union[None, Tuple[int, Union[int, float]], Tuple[int, None]]:
    if type is None:
        return None
    else:
        assert type in IMAGE_WRITE_FLAG_DICT, AttributeError(f"Invalid image type {type}")

    flag: int = IMAGE_WRITE_FLAG_DICT[type]
    if type in ['webp', 'jpeg']:
        if quality is not None:
            return flag, quality
    return flag, None


def write_image_to_file(
    image: np.ndarray,
    filename: Union[str, Path],
    *,
    type: Optional[ImageWriteFlag] = None,
    quality: int = 95,
) -> None:
    """Write image to file.

    Args:
        image: Image to be written
        filename: Path to the image
        type:
            Type of the image
            (e.g. 'png', 'jpg', 'webp', default is None, meaning the type is inferred from the filename)
        quality: Quality of the image (only valid for JPEG, default is 95)

    Raises:
        AttributeError: If the image cannot be written

    Examples:
        >>> from pythoncv.io import write_image_to_file
        >>> write_image_to_file(image, 'path/to/image.jpg')

    See Also:
        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    """
    if type is not None:
        flag = _image_write_flag_wrapper(type, quality)
    else:
        flag = None
    result = cv2.imwrite(str(filename), image[..., ::-1], flag)
    assert result, AttributeError(f"Failed to write image to {os.path.abspath(filename)}")


def write_image_to_bytes(
    image: np.ndarray,
    *,
    type: Optional[ImageWriteFlag] = None,
    quality: int = 95,
) -> bytes:
    """Write image to bytes.

    Args:
        image: Image to be written
        type:
            Type of the image
            (e.g. 'png', 'jpg', 'webp', default is None, meaning the type is inferred from the filename)
        quality: Quality of the image (only valid for JPEG, default is 95)

    Raises:
        AttributeError: If the image cannot be written

    Examples:
        >>> from pythoncv.io import write_image_to_bytes
        >>> image_bytes = write_image_to_bytes(image)

    See Also:
        https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63
    """
    flag = _image_write_flag_wrapper(type, quality)
    ret, result = cv2.imencode('.jpg', image[..., ::-1], flag)
    assert ret, AttributeError("Failed to write image to bytes")
    return result.tobytes()
