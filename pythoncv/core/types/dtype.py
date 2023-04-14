from typing import Literal

import cv2  # type: ignore

CV_DTYPE_DICT = {
    'uint8': cv2.CV_8U,
    'int8': cv2.CV_8S,
    'uint16': cv2.CV_16U,
    'int16': cv2.CV_16S,
    'int32': cv2.CV_32S,
    'float32': cv2.CV_32F,
    'float64': cv2.CV_64F,
}

CVDType = Literal['uint8', 'int8', 'uint16', 'int16', 'int32', 'float32', 'float64']
