import pytest
from pythoncv.io.video import *


def test_read_video_form_device():
    """Test read video from device."""
    video = read_video_from_device(0, backend='d-show')
    assert isinstance(video, Video)
    with pytest.raises(ValueError):
        len(video)

    for frame in video:
        assert frame is not None
        assert frame.shape == (video.info.frame_height, video.info.frame_width, 3)
        assert frame.dtype == np.uint8
        assert frame.shape[:2] != (0, 0)
        break

    video.info.fps = 10
    assert video.info.fps == 10
    assert video._cap.get(cv2.CAP_PROP_FPS) == 10

    video.info.frame_width = 1280
    video.info.frame_height = 720
    assert video.info.frame_width == 1280
    assert video._cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    assert video.info.frame_height == 720
    assert video._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720


def test_video_exception():
    video = read_video_from_device(0)

    for frame in video:
        assert frame is not None
        assert frame.shape == (video.info.frame_height, video.info.frame_width, 3)
        assert frame.dtype == np.uint8
        assert frame.shape[:2] != (0, 0)
        break

    with pytest.raises(RuntimeError):
        video.info.fps = 10
