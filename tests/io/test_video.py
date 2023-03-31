import pytest
from pythoncv.io.video import *
from tests.utils import *


@pytest.mark.skipif(check_in_ci(), reason='Skip in CI. Need a camera.')
def test_read_video_form_device():
    """Test read video from device."""
    video = read_video_from_device(0, backend='d-show')
    assert video.info.frame_count is None
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


@pytest.mark.skipif(check_in_ci(), reason='Skip in CI. Need a camera.')
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
    ori_video = read_video_from_file('demos/sample.mp4')
    assert len(tmp_video) == len(ori_video)
    assert tmp_video.info.fps == video.info.fps
    assert tmp_video.info.frame_height == video.info.frame_height
    assert tmp_video.info.frame_width == video.info.frame_width


def test_video_properties():
    video = read_video_from_file('demos/sample.mp4')
    cap = cv2.VideoCapture('demos/sample.mp4')

    assert video.info.fps == cap.get(cv2.CAP_PROP_FPS)
    assert video.info.frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert video.info.frame_height == cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert video.info.frame_width == cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    assert video.fps == cap.get(cv2.CAP_PROP_FPS)
    assert len(video) == cap.get(cv2.CAP_PROP_FRAME_COUNT)


@pytest.mark.skipif(not cv2.VideoCapture(0).isOpened(), reason="No camera device found.")
def test_device_properties():
    video = read_video_from_device(0, backend='d-show')

    video.info.frame_width = 1280
    video.info.frame_height = 720

    assert video.info.frame_width == 1280
    assert video.info.frame_height == 720
    assert video._cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    assert video._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720


def test_illegal_video_info_properties():
    video = read_video_from_file('demos/sample.mp4')

    assert video.info.fps != 0
    assert video.fps == video.info.fps == video._cap.get(cv2.CAP_PROP_FPS)
    assert video.fps == 1/video.wait_time

    with pytest.raises(RuntimeError):
        video.info.frame_width = 1280

    with pytest.raises(AttributeError):
        video.info.xyz = 1280

    with pytest.raises(AttributeError):
        print(video.info.xyz)


def test_video_writer_info():
    import tempfile
    tmp_path = tempfile.mktemp(suffix='.mp4')
    video = read_video_from_file('demos/sample.mp4')
    writer = VideoWriter(tmp_path, video.fps, (video.info.frame_height, video.info.frame_width), "mp4v")
    assert writer.info.n_frames is not None


def test_illegal_video_function_params(monkeypatch):
    with pytest.raises(FileNotFoundError):
        read_video_from_file(1)

    with pytest.raises(FileNotFoundError):
        read_video_from_file(object())

    with pytest.raises(TypeError):
        read_video_from_device('asd')
