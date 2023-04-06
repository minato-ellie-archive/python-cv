""" Blur functions.

This module contains functions for blurring images.
Every function in this module is like
def function_name(x: np.ndarray, ...) -> np.ndarray:

Notes:
    Image in pythoncv shoule be a numpy.ndarray object, which has the shape of (height, width, channel).
    No matter what the order of the channel is.

"""
import warnings
from typing import Tuple, Union

import cv2  # type: ignore
import numpy as np

from pythoncv.types.filter import BORDER_TYPES_DICT, BorderTypes


def _copy_if_not_inplace(x: np.ndarray, inplace: bool) -> np.ndarray:
    if not inplace:
        return x.copy()
    return x


def box_blur(
    x: np.ndarray,
    ksize: Tuple[int, int] = (3, 3),
    anchor: Tuple[int, int] = (-1, -1),
    normalize: bool = True,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Blurs an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \\alpha \\begin{bmatrix}
        1 & 1 & 1 & ... & 1 \\\
        1 & 1 & 1 & ... & 1 \\\
        1 & 1 & 1 & ... & 1 \\\
        ... & ... & ... & ... & ... \\\
        1 & 1 & 1 & ... & 1
        \\end{bmatrix}

    where :math:`\\alpha = \\frac{1}{ksize.width*ksize.height}`, when the parameter normalize is true, and
    :math:`\\alpha = 1` is used when the parameter normalize is false.

    Unnormalized box filter is useful for computing various integral characteristics over each pixel neighborhood,
    such as covariance matrices of image derivatives (used in dense optical flow algorithms, and so on).
    If you need to compute pixel sums over variable-size windows, use integral.

    Args:
        x: input image.
        ksize: blurring kernel size.
        anchor: 	anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
        normalize: specifying whether the kernel is normalized by its area or not.
        border_type: border mode used to extrapolate pixels outside the image.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gad533230ebf2d42509547d514f7d3fbc3)
    """
    dst = _copy_if_not_inplace(x, inplace)
    return cv2.boxFilter(x, -1, ksize, dst, anchor, normalize, BORDER_TYPES_DICT[border_type])


def blur(
    x: np.ndarray,
    ksize: Tuple[int, int] = (3, 3),
    anchor: Tuple[int, int] = (-1, -1),
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Blurs an image using the normalized box filter.(alias of box_blur)

    The function smooths an image using the kernel:

    .. math::
        K = \\frac{1}{ksize.width*ksize.height}
        \\begin{bmatrix}
                1 & 1 & 1 & ... & 1 \\\
                1 & 1 & 1 & ... & 1 \\\
                1 & 1 & 1 & ... & 1 \\\
                ... & ... & ... & ... & ... \\\
                1 & 1 & 1 & ... & 1
        \\end{bmatrix}

    Args:
        x:
            input image.
            it can have any number of channels, which
            are processed independently, but the depth
            should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
        ksize: blurring kernel size.
        anchor:
            anchor point.
            default value Point(-1,-1) means that the anchor is at the kernel center.
        border_type: border mode used to extrapolate pixels outside the image.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)
    """
    dst = _copy_if_not_inplace(x, inplace)
    return cv2.blur(x, ksize, dst, anchor, BORDER_TYPES_DICT[border_type])


def gaussian_blur(
    x: np.ndarray,
    ksize: Tuple[int, int] = (3, 3),
    sigma_x: float = 0,
    sigma_y: float = 0,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    """ Blurs an image using a Gaussian filter.

    Args:
        x: input image.
        ksize:
            Gaussian kernel size.
            ksize.width and ksize.height can differ but they both must be positive and odd.
            Or, they can be zero's, and then they are computed from sigma.
        sigma_x: Gaussian kernel standard deviation in X direction.
        sigma_y: Gaussian kernel standard deviation in Y direction.
        border_type: pixel extrapolation method
        inplace: if True, the input image will be modified inplace.

    Notes:
        if sigmaY is zero, it is set to be equal to sigmaX,
        if both sigmas are zeros, they are computed from ksize.width and ksize.height,
        respectively (see getGaussianKernel for details) to fully control the result
        regardless of possible future modifications of all this semantics, it is
        recommended to specify all of ksize, sigmaX, and sigmaY.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)

    """
    dst = _copy_if_not_inplace(x, inplace)
    return cv2.GaussianBlur(x, ksize, sigma_x, dst, sigma_y, BORDER_TYPES_DICT[border_type])


def median_blur(
    x: np.ndarray,
    ksize: int = 3,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """ Blurs an image using the median filter.

    The function smoothes an image using the median filter with the :math:`ksize \\times ksize` aperture.
    Each channel of a multi-channel image is processed independently. In-place operation is supported.



    Args:
        x: input image.(channel must be 1, 3 or 4)
        ksize: aperture linear size; it must be odd and greater than 1.(e.g. 3, 5, 7 ...)
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
    """
    dst = _copy_if_not_inplace(x, inplace)
    return cv2.medianBlur(x, ksize, dst)


def bilateral_filter(
    x: np.ndarray,
    d: int = 5,
    sigma_color: float = 75,
    sigma_space: float = 75,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
):
    """ Applies the bilateral filter to an image.

    The function applies bilateral filtering to the input image, as described in
    [Bilateral Filtering for Gray and Color Images](http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html).

    bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp.
    However, it is very slow compared to most filters.

    Args:
        x: input image.(channel must be 1 or 3, and must be 8bit or 32bit float)
        d:
            Diameter of each pixel neighborhood that is used during filtering.
            If it is non-positive, it is computed from `sigma_space`.
        sigma_color:
            Filter sigma in the color space.
            A larger value of the parameter means that farther colors within the pixel neighborhood
            will be mixed together, resulting in larger areas of semi-equal color.
        sigma_space:
        Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will
        influence each other as long as their colors are close enough .
        When d>0, it specifies the neighborhood size regardless of sigmaSpace.
        Otherwise, d is proportional to sigmaSpace
        border_type: border mode used to extrapolate pixels outside the image.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

        [Bilateral Filtering for Gray and Color Images](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html)

    """  # noqa: E501
    if inplace:
        raise NotImplementedError("inplace is not supported for bilateral_filter")

    dst = _copy_if_not_inplace(x, inplace)
    return cv2.bilateralFilter(x, d, sigma_color, sigma_space, dst, BORDER_TYPES_DICT[border_type])


def stack_blur(
        x: np.ndarray,
        ksize: Union[Tuple[int, int], int] = (3, 3),
        *,
        inplace: bool = False,
) -> np.ndarray:
    """ Blurs an image using the stackBlur.

    The function uses stackBlur,
    a fast alternative to Gaussian blur that doesn't take longer as the blur size increases, to process an image.
    It works by creating a moving stack of colors while scanning the image.
    New blocks of color are added to the right side while the leftmost color is removed.
    The colors on the top layer of the stack are either added or reduced by one,
    depending on their position relative to the stack.
    The function only supports BORDER_REPLICATE as the border type.

    Notes:
        Original paper was proposed by Mario Klingemann,
        which can be found http://underdestruction.com/2004/02/25/stackblur-2004.

    Args:
        x: input image.
        ksize:
            stack-blurring kernel size.
            The ksize.width and ksize.height can differ but they both must be positive and odd.
            (e.g. (3, 3), (3, 5), (5, 3), (5, 5) ...)
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga13a01048a8a200aab032ce86a9e7c7be)

    """
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    dst = _copy_if_not_inplace(x, inplace)
    return cv2.stackBlur(x, ksize, dst)


def square_blur(
    x: np.ndarray,
    ksize: Union[Tuple[int, int], int] = (3, 3),
    anchor: Tuple[int, int] = (-1, -1),
    normalize: bool = True,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    """ Calculates the normalized sum of squares of the pixel values overlapping the filter.

    Warnings:
        This function is not currently supported.

    Args:
        x: input image.
        ksize: kernel size.
        anchor: anchor point; default value (-1, -1) means that the anchor is at the kernel center.
        normalize: specifying whether the kernel is to be normalized by its area or not.
        border_type: border mode used to extrapolate pixels outside the image.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga76e863e7869912edbe88321253b72688)
    """
    if inplace:
        raise NotImplementedError("inplace is not supported for square_blur")

    warnings.warn("This function is not currently supported by OpenCV.", RuntimeWarning)
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    dst = _copy_if_not_inplace(x, inplace)
    return cv2.sqrBoxFilter(x, -1, ksize, dst, anchor, normalize, BORDER_TYPES_DICT[border_type])
