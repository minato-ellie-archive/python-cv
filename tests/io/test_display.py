import pytest
import pytest_mock as pm
import cv2
from pythoncv.io.display import *
from tests.utils import *


def patch_cv2(mocker: pm.MockerFixture):
    mocker.patch.object(cv2, 'namedWindow')
    mocker.patch.object(cv2, 'resizeWindow')
    mocker.patch.object(cv2, 'destroyWindow')
    mocker.patch.object(cv2, 'getWindowImageRect')
    mocker.patch.object(cv2, 'getWindowProperty')
    mocker.patch.object(cv2, 'setWindowProperty')
    mocker.patch.object(cv2, 'resizeWindow')


def test_display_video(mocker):
    """Test display video."""
    patch_cv2(mocker)

    window = VideoWindow(name='test', size=(640, 480), type='normal')
    cv2.namedWindow.call_count == 0
    cv2.resizeWindow.call_count == 0
    window.open()

    cv2.namedWindow.assert_called_once_with('test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow.assert_called_once_with('test', 640, 480)

    _ = window.aspect_ratio
    cv2.getWindowProperty.assert_called_once_with('test', cv2.WND_PROP_ASPECT_RATIO)
    for _ in range(10):
        window.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    window.close()
    cv2.destroyWindow.assert_called_once_with('test')


def test_display_video_with_context_manager(mocker):
    """Test display video with context manager."""
    patch_cv2(mocker)
    with VideoWindow(name='test', size=(640, 480), type='normal') as window:
        _ = window.size
        cv2.getWindowImageRect.assert_called_once_with('test')
        _ = window.aspect_ratio
        cv2.getWindowProperty.assert_called_once_with('test', cv2.WND_PROP_ASPECT_RATIO)
        for _ in range(10):
            window.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        cv2.destroyWindow.assert_not_called()

    cv2.destroyWindow.assert_called_once_with('test')

    with pytest.raises(RuntimeError):
        del window
        window = VideoWindow(name='test', size=(640, 480), type='normal')
        window.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))


@pytest.mark.skipif(check_in_ci(), reason="Skip in CI. Can not open window with certainty monitor size.")
def test_display_attribute():
    """Test display attribute."""
    window = VideoWindow(name='test', size=(640, 480), type='normal')
    window.open()
    assert window.size == (640, 480)
    assert window.aspect_ratio == 1.3333333333333333
    window.size = (480, 640)
    assert window.size == (480, 640)
    assert window.aspect_ratio == 0.75
    window.aspect_ratio = "freeratio"
    window.aspect_ratio = "keepratio"
    with pytest.raises(ValueError):
        window.aspect_ratio = "test"
    window.close()

    window = VideoWindow(name='test', size=(640, 480), type='normal')
    assert isinstance(window.opengl, bool)
    assert (cv2.getWindowProperty(window.name, cv2.WND_PROP_OPENGL) == 1) == window.opengl

    window.open()
    assert isinstance(window.visible, bool)
    assert (cv2.getWindowProperty(window.name, cv2.WND_PROP_VISIBLE) == 1) == window.visible
    window.close()
    assert window.visible is False

    window = VideoWindow(name='test', size=(640, 480), type='normal')
    with pytest.raises(RuntimeError):
        assert window.topmost

    with pytest.raises(RuntimeError):
        assert window.vsync
    window.open()
    assert isinstance(window.topmost, bool)
    assert (cv2.getWindowProperty(window.name, cv2.WND_PROP_TOPMOST) == 1) == window.topmost

    window.open()
    assert isinstance(window.vsync, bool)

    del window
    window = VideoWindow(name='test', size=(640, 480), type='normal')
    with pytest.raises(RuntimeError):
        assert window.auto_size
    window.open()
    assert (cv2.getWindowProperty(window.name, cv2.WND_PROP_AUTOSIZE) == 1) == window.auto_size

    del window
    window = VideoWindow(name='test', size=(640, 480), type='normal')
    window.open()
    assert window.size == (640, 480)
    window.size = (480, 640)
    assert window.size == (480, 640)

    del window
    window = VideoWindow(name='test', size=(640, 480), type='normal')
    window.open()
    assert window.fullscreen is False
    window.fullscreen = True
    assert window.fullscreen is True
    window.fullscreen = False
    assert window.fullscreen is False


@pytest.mark.skipif(check_in_ci(), reason="Skip in CI. Need a display to run this test.")
def test_display_repr():
    """Test display repr."""
    window = VideoWindow(name='test', size=(640, 480), type='normal')
    assert repr(window) == "VideoWindow(name='test', size=(640, 480), is_open=False)"

    window.open()
    assert repr(window) == "VideoWindow(name='test', size=(640, 480), is_open=True)"

    window.close()
    assert repr(window) == "VideoWindow(name='test', size=(640, 480), is_open=False)"

    assert str(window) == "VideoWindow(name='test', size=(640, 480), is_open=False)"
