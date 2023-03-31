import pytest
from pythoncv.io.display import *
from tests.utils import *


@pytest.mark.skipif(check_in_ci(), reason='Skip in CI. Display is not available.')
def test_display_video():
    """Test display video."""
    window = VideoWindow(name='test', size=(640, 480), type='normal')
    window.open()
    assert window.size == (640, 480)
    assert window.aspect_ratio == 1.3333333333333333
    for _ in range(10):
        window.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    window.close()

    with VideoWindow(name='test', size=(640, 480), type='normal') as window:
        assert window.size == (640, 480)
        assert window.aspect_ratio == 1.3333333333333333
        for _ in range(10):
            window.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

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
