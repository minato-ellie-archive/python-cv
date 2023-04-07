from typing import Literal

import cv2  # type: ignore

__all__ = [
    'WINDOW_FLAGS_DICT',
    'WINDOW_FLAGS_INVERSE_DICT',
    'WindowFlags',
]

WINDOW_FLAGS_DICT = {
    'normal': cv2.WINDOW_NORMAL,
    'autosize': cv2.WINDOW_AUTOSIZE,
    'opengl': cv2.WINDOW_OPENGL,
    'fullscreen': cv2.WINDOW_FULLSCREEN,
    'freeratio': cv2.WINDOW_FREERATIO,
    'keepratio': cv2.WINDOW_KEEPRATIO,
    'gui_expanded': cv2.WINDOW_GUI_EXPANDED,
    'gui_normal': cv2.WINDOW_GUI_NORMAL,
}

WINDOW_FLAGS_INVERSE_DICT = {v: k for k, v in WINDOW_FLAGS_DICT.items()}

WindowFlags = Literal['normal', 'autosize', 'opengl', 'fullscreen', 'freeratio', 'keepratio', 'gui_expanded',
                      'gui_normal',]
