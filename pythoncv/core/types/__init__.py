""" Type definitions.

This module contains type definitions for PythonCV.

"""

from .video import (
    VideoCaptureProperties,
    VideoWriterProperties,
    CAPTURE_BACKEND_DICT,
    CaptureBackends,
    FourCC,
)
from .display import (
    WINDOW_FLAGS_DICT,
    WINDOW_FLAGS_INVERSE_DICT,
    WindowFlags,
)
from .filter import (BORDER_TYPES_DICT, BorderTypes)
from .image import (IMAGE_READ_FLAG_DICT, ImageReadFlags, IMAGE_WRITE_FLAG_DICT, ImageWriteFlags)
