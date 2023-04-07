import pytest

from pythoncv.functions.blur import *


def test_box_blur(mocker):
    mocker.spy(cv2, 'boxFilter')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = box_blur(arr)

    cv2.boxFilter.assert_called_once()
    assert not np.allclose(arr, result)

    result = box_blur(arr, ksize=(3, 3), inplace=True)
    assert np.allclose(arr, result)
    cv2.boxFilter.assert_called_with(arr, -1, (3, 3), arr, (-1, -1), True, cv2.BORDER_REFLECT_101)


def test_blur(mocker):
    mocker.spy(cv2, 'blur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = blur(arr)

    cv2.blur.assert_called_once()
    assert not np.allclose(arr, result)

    result = blur(arr, ksize=(3, 3), inplace=True)
    assert np.allclose(arr, result)
    cv2.blur.assert_called_with(arr, (3, 3), arr, (-1, -1), cv2.BORDER_REFLECT_101)


def test_gaussian_blur(mocker):
    mocker.spy(cv2, 'GaussianBlur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = gaussian_blur(arr)

    cv2.GaussianBlur.assert_called_once()
    assert not np.allclose(arr, result)

    result = gaussian_blur(arr, ksize=(3, 3), inplace=True)
    assert np.allclose(arr, result)
    cv2.GaussianBlur.assert_called_with(arr, (3, 3), 0, arr, 0, cv2.BORDER_REFLECT_101)


def test_median_blur(mocker):
    mocker.spy(cv2, 'medianBlur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = median_blur(arr)

    cv2.medianBlur.assert_called_once()
    assert not np.allclose(arr, result)

    result = median_blur(arr, ksize=3, inplace=True)
    assert np.allclose(arr, result)
    cv2.medianBlur.assert_called_with(arr, 3, arr)


def test_bilateral_filter(mocker):
    mocker.spy(cv2, 'bilateralFilter')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = bilateral_filter(arr)

    cv2.bilateralFilter.assert_called_once()
    assert not np.allclose(arr, result)

    with pytest.raises(NotImplementedError):
        result = bilateral_filter(arr, d=5, inplace=True)


def test_stack_blur(mocker):
    mocker.spy(cv2, 'stackBlur')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = stack_blur(arr)

    cv2.stackBlur.assert_called_once()
    ori = cv2.stackBlur.call_args[0][0]
    dst = cv2.stackBlur.call_args[0][2]
    assert ori is not dst
    assert not np.allclose(arr, result)

    result = stack_blur(arr, ksize=3, inplace=True)
    assert np.allclose(arr, result)
    cv2.stackBlur.assert_called_with(arr, (3, 3), arr)
    ori = cv2.stackBlur.call_args[0][0]
    dst = cv2.stackBlur.call_args[0][2]
    assert ori is dst


def test_square_blur(mocker):
    mocker.spy(cv2, 'sqrBoxFilter')
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = square_blur(arr)

    cv2.sqrBoxFilter.assert_called_once()
    assert not np.allclose(arr, result)

    result = square_blur(arr, ksize=3)
    assert cv2.sqrBoxFilter.call_count == 2
    assert cv2.sqrBoxFilter.call_args_list[0][0][2] == cv2.sqrBoxFilter.call_args_list[1][0][2]

    with pytest.raises(NotImplementedError):
        result = square_blur(arr, ksize=3, inplace=True)
