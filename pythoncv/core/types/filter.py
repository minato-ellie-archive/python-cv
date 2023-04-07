from typing import Literal

import cv2  # type: ignore

__all__ = [
    'BORDER_TYPES_DICT',
    'BorderTypes',
]

BORDER_TYPES_DICT = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT_101,
    'wrap': cv2.BORDER_WRAP,
    'isolated': cv2.BORDER_ISOLATED,
    'default': cv2.BORDER_DEFAULT,
}

BorderTypes = Literal['constant', 'replicate', 'reflect', 'reflect101', 'wrap', 'isolated', 'default']
