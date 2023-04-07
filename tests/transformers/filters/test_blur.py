import cv2  # type: ignore
import numpy as np
import pytest

from pythoncv.transformers.filters.blur import *


def test_box_blur_filter(mocker):
    mocker.spy(f, 'box_blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = box_blur()
    assert f.box_blur.call_count == 0
    result = fn(arr)
    f.box_blur.assert_called_once()

    # Default parameters
    ref = cv2.boxFilter(arr, -1, (3, 3), arr, (-1, -1), True, cv2.BORDER_REFLECT_101)
    assert np.allclose(ref, result)


def test_blur_filter(mocker):
    mocker.spy(f, 'blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = blur()
    assert f.blur.call_count == 0
    result = fn(arr)
    f.blur.assert_called_once()

    # Default parameters
    ref = cv2.blur(arr, (3, 3), arr, (-1, -1), cv2.BORDER_REFLECT_101)
    assert np.allclose(ref, result)


def test_gaussian_blur_filter(mocker):
    mocker.spy(f, 'gaussian_blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = gaussian_blur()
    assert f.gaussian_blur.call_count == 0
    result = fn(arr)
    f.gaussian_blur.assert_called_once()

    # Default parameters
    ref = cv2.GaussianBlur(arr, (3, 3), 0, arr, 0, cv2.BORDER_REFLECT_101)
    assert np.allclose(ref, result)


def test_median_blur_filter(mocker):
    mocker.spy(f, 'median_blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = median_blur()
    assert f.median_blur.call_count == 0
    result = fn(arr)
    f.median_blur.assert_called_once()

    # Default parameters
    ref = cv2.medianBlur(arr, 3, arr)
    assert np.allclose(ref, result)


def test_bilateral_filter_filter(mocker):
    mocker.spy(f, 'bilateral_filter')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = bilateral_filter()
    assert f.bilateral_filter.call_count == 0
    result = fn(arr)
    f.bilateral_filter.assert_called_once()


def test_stack_blur_filter(mocker):
    mocker.spy(f, 'stack_blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = stack_blur()
    assert f.stack_blur.call_count == 0
    result = fn(arr)
    f.stack_blur.assert_called_once()

    # Default parameters
    ref = cv2.stackBlur(arr, (3, 3))
    assert np.allclose(ref, result)


def test_square_blur_filter(mocker):
    mocker.spy(f, 'square_blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fn = square_blur()
    assert f.square_blur.call_count == 0
    result = fn(arr)
    f.square_blur.assert_called_once()

    # Default parameters
    ref = cv2.sqrBoxFilter(arr, -1, (3, 3))
    assert np.allclose(ref, result)


def test_illegal_box_filter_parameters():
    with pytest.raises(TypeError):
        fn = box_blur(ksize='3')

    with pytest.raises(TypeError):
        fn = box_blur(anchor='3')

    with pytest.raises(TypeError):
        fn = box_blur(normalize=18)

    with pytest.raises(ValueError):
        fn = box_blur(ksize=[3, 3, 3])

    fn = box_blur(ksize=[3, 3])
    ref_fn = box_blur(ksize=3)

    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    assert np.allclose(ref_fn(arr), fn(arr))
