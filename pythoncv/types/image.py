from typing import Literal

import cv2  # type: ignore

IMAGE_READ_FLAG_DICT = {
    'unchanged': cv2.IMREAD_UNCHANGED,
    'grayscale': cv2.IMREAD_GRAYSCALE,
    'color': cv2.IMREAD_COLOR,
    'anydepth': cv2.IMREAD_ANYDEPTH,
    'anycolor': cv2.IMREAD_ANYCOLOR,
    'load_gdal': cv2.IMREAD_LOAD_GDAL,
    'ignore_orientation': cv2.IMREAD_IGNORE_ORIENTATION,
    ('grayscale', 2): cv2.IMREAD_REDUCED_GRAYSCALE_2,
    ('grayscale', 4): cv2.IMREAD_REDUCED_GRAYSCALE_4,
    ('grayscale', 8): cv2.IMREAD_REDUCED_GRAYSCALE_8,
    ('color', 2): cv2.IMREAD_REDUCED_COLOR_2,
    ('color', 4): cv2.IMREAD_REDUCED_COLOR_4,
    ('color', 8): cv2.IMREAD_REDUCED_COLOR_8,
}

ImageReadFlag = Literal['unchanged', 'grayscale', 'color',]

IMAGE_WRITE_FLAG_DICT = {
    'jpeg': cv2.IMWRITE_JPEG_QUALITY,
    'png': cv2.IMWRITE_PNG_COMPRESSION,
    'webp': cv2.IMWRITE_WEBP_QUALITY,
    'tiff': cv2.IMWRITE_TIFF_COMPRESSION,
    'exr': cv2.IMWRITE_EXR_TYPE,
}

ImageWriteFlag = Literal['jpeg', 'png', 'webp', 'tiff', 'exr',]
