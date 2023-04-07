import pytest
from PIL import Image
from pythoncv.io.image import *


def test_read_image_from_file():
    image = read_image_from_file('demos/sample.jpg')
    assert image.shape[2] == 3
    assert isinstance(image, np.ndarray)
    assert type(image) == CVImage

    image = read_image_from_file('demos/sample.jpg')
    ref_image = np.array(Image.open('demos/sample.jpg'))
    assert isinstance(image, np.ndarray)
    assert type(image) == CVImage

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
        assert isinstance(image, np.ndarray)
        assert type(image) == CVImage

    with open('demos/sample.jpg', 'rb') as f:
        ref_image = np.array(Image.open('demos/sample.jpg'))

    assert np.all(image == ref_image)

    with open('demos/sample.jpg', 'rb') as f:
        image = read_image_from_bytes(f.read(), color_mode='grayscale')
        assert len(image.shape) == 2
        assert isinstance(image, np.ndarray)
        assert type(image) == CVImage

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
    assert isinstance(image, np.ndarray)
    assert type(image) == CVImage

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


def test_write_image_to_file():
    import tempfile

    image = read_image('demos/sample.jpg')
    with tempfile.TemporaryDirectory() as tmpdir:
        write_image_to_file(image, tmpdir + '/sample.jpg')
        image = read_image(tmpdir + '/sample.jpg')
        assert image.shape[2] == 3

        ref_image = np.array(Image.open('demos/sample.jpg'))
        assert np.allclose(image, ref_image, atol=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_image_to_file(image, tmpdir + '/sample.png', type='png')
        image = read_image(tmpdir + '/sample.png')
        assert image.shape[2] == 3
        assert np.allclose(image, ref_image, atol=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_image_to_file(image, tmpdir + '/sample.jpg', type='jpeg', quality=50)
        image = read_image(tmpdir + '/sample.jpg')
        assert image.shape[2] == 3
        assert os.path.exists(tmpdir + '/sample.jpg')
        assert os.path.getsize(tmpdir + '/sample.jpg') < os.path.getsize('demos/sample.jpg')


def test_write_image_to_bytes():
    image = read_image('demos/sample.jpg')
    image_bytes = write_image_to_bytes(image)
    image = read_image_from_bytes(image_bytes)
    assert image.shape[2] == 3

    ref_image = np.array(Image.open('demos/sample.jpg'))
    assert np.allclose(image, ref_image, atol=10)

    image_bytes = write_image_to_bytes(image, type='png')
    image = read_image_from_bytes(image_bytes)
    assert image.shape[2] == 3
    assert np.allclose(image, ref_image, atol=20)

    image_bytes = write_image_to_bytes(image, type='jpeg', quality=50)
    image = read_image_from_bytes(image_bytes)
    assert image.shape[2] == 3
    assert len(image_bytes) < os.path.getsize('demos/sample.jpg')
