import pytest
from PIL import Image
from pythoncv.io.image import *


def test_read_image_from_file():
    image = read_image_from_file('demos/sample.jpg')
    assert image.shape[2] == 3

    image = read_image_from_file('demos/sample.jpg')
    ref_image = np.array(Image.open('demos/sample.jpg'))

    assert np.all(image == ref_image)

    image = read_image_from_file('demos/sample.jpg', color_mode='grayscale')
    assert len(image.shape) == 2
    with pytest.raises(AttributeError):
        image = read_image_from_file('demos/sample.jpg', reduce_ratio=2)

    image = read_image_from_file('demos/sample.jpg', color_mode='color', reduce_ratio=4)

    assert image.shape[0] == ref_image.shape[0] // 4
    assert image.shape[1] == ref_image.shape[1] // 4

    with pytest.raises(AttributeError):
        image = read_image_from_file('demos/sample.jpg', color_mode='unchanged', reduce_ratio=8)


def test_read_image_from_bytes():
    with open('demos/sample.jpg', 'rb') as f:
        image = read_image_from_bytes(f.read())
        assert image.shape[2] == 3

    with open('demos/sample.jpg', 'rb') as f:
        ref_image = np.array(Image.open('demos/sample.jpg'))

    assert np.all(image == ref_image)

    with open('demos/sample.jpg', 'rb') as f:
        image = read_image_from_bytes(f.read(), color_mode='grayscale')
        assert len(image.shape) == 2

    with open('demos/sample.jpg', 'rb') as f:
        image = read_image_from_bytes(f.read(), color_mode='color', reduce_ratio=4)

    assert image.shape[0] == ref_image.shape[0] // 4
    assert image.shape[1] == ref_image.shape[1] // 4

    with pytest.raises(AttributeError):
        with open('demos/sample.jpg', 'rb') as f:
            image = read_image_from_bytes(f.read(), color_mode='unchanged', reduce_ratio=8)


def test_read_image():
    image = read_image('demos/sample.jpg')
    assert image.shape[2] == 3

    ref_image = np.array(Image.open('demos/sample.jpg'))

    assert np.all(image == ref_image)

    image = read_image('demos/sample.jpg', color_mode='grayscale')
    assert len(image.shape) == 2

    image = read_image('demos/sample.jpg', color_mode='color', reduce_ratio=4)

    assert image.shape[0] == ref_image.shape[0] // 4
    assert image.shape[1] == ref_image.shape[1] // 4

    with pytest.raises(AttributeError):
        image = read_image('demos/sample.jpg', color_mode='unchanged', reduce_ratio=8)

    with open('demos/sample.jpg', 'rb') as f:
        image = read_image(f.read(), color_mode='color', reduce_ratio=4)

    assert image.shape[0] == ref_image.shape[0] // 4
    assert image.shape[1] == ref_image.shape[1] // 4

    with pytest.raises(AttributeError):
        with open('demos/sample.jpg', 'rb') as f:
            image = read_image(f.read(), color_mode='unchanged', reduce_ratio=8)
