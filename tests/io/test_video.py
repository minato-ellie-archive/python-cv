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

    video.fps = 50
    assert video.info.fps == 50
    assert video.fps == 50
    assert video._cap.get(cv2.CAP_PROP_FPS) == 50

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

    with pytest.raises(RuntimeError):
        video.fps = 50


def test_read_video_from_file():
    """Test read video from file."""
    video = read_video_from_file('demos/sample.mp4')
    assert isinstance(video, Video)
    assert len(video) > 0 and isinstance(len(video), int)

    count = 0
    for idx, frame in enumerate(video):
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (video.info.frame_height, video.info.frame_width, 3)
        assert frame.dtype == np.uint8
        assert frame.shape[:2] != (0, 0)
        count += 1
        if idx == 10:
            break

    assert count == 11

    assert video.info.fps > 10


def test_read_video_from_file_exception():
    """Test read video from file exception."""
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        read_video_from_file('demos/sample.mp5')

    with pytest.raises(Exception):
        read_video_from_file(object())

    video = read_video_from_file(Path('demos/sample.mp4'))

    for frame in video:
        assert frame is not None
        assert frame.shape == (video.info.frame_height, video.info.frame_width, 3)
        assert frame.dtype == np.uint8
        assert frame.shape[:2] != (0, 0)
        break


def test_read_video_from_url():
    """Test read video from url."""
    video = read_video_from_url(
        'https://mazwai.com/videvo_files/video/free/2018-12/small_watermarked/180607_A_124_preview.mp4')  # noqa: E501
    assert isinstance(video, Video)
    assert len(video) > 0 and isinstance(len(video), int)

    count = 0
    for idx, frame in enumerate(video):
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (video.info.frame_height, video.info.frame_width, 3)
        assert frame.dtype == np.uint8
        assert frame.shape[:2] != (0, 0)
        count += 1
        if idx == 10:
            break

    assert count == 11


def test_read_video_from_url_exception():
    """Test read video from url."""
    with pytest.raises(TypeError):
        read_video_from_url(1)

    video = read_video_from_url(
        'https://mazwai.com/videvo_files/video/free/2018-12/small_watermarked/180607_A_124_preview.mp4')  # noqa: E501
    assert isinstance(video, Video)
    assert len(video) > 0 and isinstance(len(video), int)

    count = 0
    for idx, frame in enumerate(video):
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (video.info.frame_height, video.info.frame_width, 3)
        assert frame.dtype == np.uint8
        assert frame.shape[:2] != (0, 0)
        count += 1
        if idx == 10:
            break

    assert count == 11

    with pytest.raises(RuntimeError):
        video.info.fps = 10

    with pytest.raises(RuntimeError):
        video.fps = 50


def test_write_video_to_file():
    """Test write video to file."""
    import tempfile

    video = read_video_from_file('demos/sample.mp4')
    tmp_path = tempfile.mktemp(suffix='.mp4')

    writer = VideoWriter(tmp_path, video.fps, (video.info.frame_height, video.info.frame_width), "mp4v")
    for frame in video:
        writer.write(frame)

    del writer

    tmp_video = read_video_from_file(tmp_path)
    assert len(tmp_video) == len(video)
    assert tmp_video.info.fps == video.info.fps
    assert tmp_video.info.frame_height == video.info.frame_height
    assert tmp_video.info.frame_width == video.info.frame_width

    for frame1, frame2 in zip(video, tmp_video):
        assert np.all(frame1 == frame2)
