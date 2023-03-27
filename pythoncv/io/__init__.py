""" PythonCV IO module.


"""

from .video import (
    BaseVideo,
    Video,
    BaseVideoWriter,
    VideoWriter,

    read_video_from_device,
    read_video_from_file,
    read_video_from_url,
)


__all__ = [
    'BaseVideo',
    'Video',
    'BaseVideoWriter',
    'VideoWriter',

    'read_video_from_device',
    'read_video_from_file',
    'read_video_from_url',

    'video',
]
