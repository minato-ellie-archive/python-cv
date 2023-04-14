""" Filter functions.

This module contains functions in the category of filter.
Every function in this module is like

```python
def function(x: np.ndarray, ..., *, inplace=False) -> np.ndarray: ...
```

Notes:
    Image in pythoncv shoule be a numpy.ndarray object, which has the shape of (height, width, channel).
    No matter what the order of the channel is.

    The channel of the image is RGB, which is different from the channel of the image in OpenCV,
    but the same as the channel of the image in PIL and Tensorflow.

    The data type of the image should be np.uint8, np.uint16, np.float32 or np.float64.

Warnings:
    In some functions, inplace is not supported.
    This time, the function will return a new array, and the original array will not be changed.
    No exception will be raised. (For Pipeline)

See Also:
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html

"""
import warnings
from typing import Tuple, Union, List, Optional

import cv2  # type: ignore
import numpy as np

from pythoncv.core.types.filter import BORDER_TYPES_DICT, BorderTypes


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
        inplace: if True, the input image will be modified inplace. (not supported)

    Warnings:
        inplace is not supported (**OpenCV does not support inplace operation for bilateralFilter**).
        [see more](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed:~:text=This%20filter%20does%20not%20work%20inplace.)

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

        [Bilateral Filtering for Gray and Color Images](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html)

    """  # noqa: E501
    if inplace:
        warnings.warn(
            "inplace is not supported for bilateralFilter. "
            "This filter does not work inplace. "
            "See more: "
            "https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed.")

    dst = x.copy()
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
        inplace: if True, the input image will be modified inplace. (not supported)

    Warnings:
        inplace is not supported (No information about inplace operation in OpenCV Doc, but it is not working).

    Returns:
        Result of the blurring operation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga76e863e7869912edbe88321253b72688)
    """
    if inplace:
        warnings.warn(
            "inplace is not supported for square_blur. "
            "This function is not currently supported by OpenCV. "
            "See more: "
            "https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga76e863e7869912edbe88321253b72688.")

    warnings.warn("This function is not currently supported by OpenCV.", RuntimeWarning)
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    dst = x.copy()
    return cv2.sqrBoxFilter(x, -1, ksize, dst, anchor, normalize, BORDER_TYPES_DICT[border_type])


def _recursion_pyrdown(
    x: np.ndarray,
    max_level: int,
    border_type: BorderTypes = "reflect101",
    level: int = 0,
) -> List[np.ndarray]:
    if level > max_level:
        return []
    return [x] + _recursion_pyrdown(cv2.pyrDown(x, borderType=BORDER_TYPES_DICT[border_type]), max_level, border_type,
                                    level + 1)


def build_pyramid(
    x: np.ndarray,
    max_level: int,
    border_type: BorderTypes = "reflect101",
) -> List[np.ndarray]:
    """ Builds a pyramid from the input image.

    Args:
        x: input image.
        max_level:
            0-based index of the last (the smallest) pyramid level.
            Len of the output list will be max_level + 1.
            Level 0 is the original image.
        border_type: border mode used to extrapolate pixels outside the image.

    Returns:
        List of images that be built by recursively applying pyrDown. Order is from the original image to the smallest.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacfdda2bc1ac55e96de7e9f0bce7238c0)
    """
    return _recursion_pyrdown(x, max_level, border_type)


def pyr_down(
    x: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Blurs an image and downsamples it.

    By default, size of the output image is computed as :math:`[\\frac{width}{2}, \\frac{height}{2}]`,
    but in any case, the following conditions should be satisfied:

    - :math:`\\texttt{dst.cols} \\leq \\texttt{src.cols}/2`
    - :math:`\\texttt{dst.rows} \\leq \\texttt{src.rows}/2`

    The function performs the downsampling step of the Gaussian pyramid construction.
    First, it convolves the source image with the kernel:

    .. math::
        \\frac{1}{256}
        \\begin{bmatrix}
            1 & 4 & 6 & 4 & 1 \\\
            4 & 16 & 24 & 16 & 4 \\\
            6 & 24 & 36 & 24 & 6 \\\
            4 & 16 & 24 & 16 & 4 \\\
            1 & 4 & 6 & 4 & 1
        \\end{bmatrix}

    Then, it downsamples the image by rejecting even rows and columns.

    Args:
        x: input image.
        output_size:
            size of the output image.
            If None, the size will be :math:`[\\frac{width}{2}, \\frac{height}{2}]`.
        border_type: border mode used to extrapolate pixels outside the image.
        inplace: This function is not supported inplace. No effect.

    Returns:
        Result of the downsampling image.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff)
    """
    if inplace:
        warnings.warn("inplace is not supported for pyr_down.", RuntimeWarning)

    x = x.copy()
    return cv2.pyrDown(x, dstsize=output_size, borderType=BORDER_TYPES_DICT[border_type])


def pyr_up(
    x: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Upsamples an image and then blurs it.

    By default, size of the output image is computed as :math:`[2 \\cdot width, 2 \\cdot height]`,
    but in any case, the following conditions should be satisfied:

    - :math:`|\\texttt{dst.width} - \\texttt{src.cols} * 2| \\leq \\texttt{dst.width} \\mod 2`
    - :math:`|\\texttt{dst.height} - \\texttt{src.rows} * 2| \\leq \\texttt{dst.height} \\mod 2`

    Args:
        x: input image.
        output_size: size of the output image.
            If None, the size will be :math:`[2 \\cdot width, 2 \\cdot height]`.
        border_type: border mode used to extrapolate pixels outside the image.
        inplace: This function is not supported inplace. No effect.

    Returns:
        Result of the upsampling image.

    See Also:
        https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gada75b59bdaaca411ed6fee10085eb784
    """
    if inplace:
        warnings.warn("inplace is not supported for pyr_up.", RuntimeWarning)

    x = x.copy()
    return cv2.pyrUp(x, dstsize=output_size, borderType=BORDER_TYPES_DICT[border_type])


def dilation(
    x: np.ndarray,
    kernel: np.ndarray,
    anchor: Tuple[int, int] = (-1, -1),
    iterations: int = 1,
    border_type: BorderTypes = "reflect101",
    border_value: int = 0,
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Dilates an image by using a specific structuring element.

    The function dilates the source image using the specified structuring element
    that determines the shape of a pixel neighborhood over which the maximum is taken:

    .. math::
        \texttt{dst} (x,y) =  \max _{(x',y'): \; \texttt{element} (x',y')  \neq 0} \texttt{src} (x+x',y+y')

    In case of multi-channel images, each channel is processed independently.

    Args:
        x: input image.
        kernel: structuring element used for dilation; if element=Mat(),
            a 3 x 3 rectangular structuring element is used.
        anchor: position of the anchor within the element; default value
            (-1, -1) means that the anchor is at the element center.
        iterations: number of times dilation is applied.
        border_type: border mode used to extrapolate pixels outside the image.
        border_value: border value in case of a constant border.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the dilation.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e78c3e1145fdd438dd86d110a62f7)
    """
    dst = _copy_if_not_inplace(x, inplace)

    return cv2.dilate(
        x,
        dst,
        kernel,
        anchor=anchor,
        iterations=iterations,
        borderType=BORDER_TYPES_DICT[border_type],
        borderValue=border_value,
    )


def erosion(
    x: np.ndarray,
    kernel: np.ndarray,
    anchor: Tuple[int, int] = (-1, -1),
    iterations: int = 1,
    border_type: BorderTypes = "reflect101",
    border_value: int = 0,
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Erodes an image by using a specific structuring element.

    The function erodes the source image using the specified structuring element
    that determines the shape of a pixel neighborhood over which the minimum is taken:

    .. math::
        \texttt{dst} (x,y) =  \min _{(x',y'): \; \texttt{element} (x',y')  \neq 0} \texttt{src} (x+x',y+y')

    In case of multi-channel images, each channel is processed independently.

    Args:
        x: input image.
        kernel: structuring element used for erosion; if element=Mat(),
            a 3 x 3 rectangular structuring element is used.
        anchor: position of the anchor within the element; default value
            (-1, -1) means that the anchor is at the element center.
        iterations: number of times erosion is applied.
        border_type: border mode used to extrapolate pixels outside the image.
        border_value: border value in case of a constant border.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the erosion.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb)
    """
    dst = _copy_if_not_inplace(x, inplace)

    return cv2.erode(
        x,
        dst,
        kernel,
        anchor=anchor,
        iterations=iterations,
        borderType=BORDER_TYPES_DICT[border_type],
        borderValue=border_value,
    )


def filter2d(
    x: np.ndarray,
    dddepth: int,
    kernel: np.ndarray,
    anchor: Tuple[int, int] = (-1, -1),
    delta: float = 0.0,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Convolves an image with the kernel.

    The function applies an arbitrary linear filter to an image. In-place operation is supported.
    When the aperture is partially outside the image, the function interpolates outlier pixel values
    according to the specified border mode.

    The function does actually compute correlation, not the convolution:

    .. math::
        \texttt{dst} (x,y) =
        \sum _{ \stackrel{0\leq x' < \texttt{kernel.cols},}{0\leq y' < \texttt{kernel.rows}} }
        \texttt{kernel} (x',y') \cdot \texttt{src} (x+x'-\texttt{anchor.x},y+y'-\texttt{anchor.y})

    That is, the kernel is not mirrored around the anchor point. If you need a real convolution,
     flip the kernel using flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1)`,

    The function uses the DFT-based algorithm in case of sufficiently large kernels ( > 11 x 11 ) and
    the direct algorithm for small kernels.

    Args:
        x: input image.
        dddepth: desired depth of the destination image.
        kernel: convolution kernel (or rather a correlation kernel), a single-channel
            floating point matrix; if you want to apply different kernels to different
            channels, split the image into separate color planes using split and process
            them individually.
        anchor: anchor of the kernel that indicates the relative position of a filtered
            point within the kernel; the anchor should lie within the kernel; default
            value (-1, -1) means that the anchor is at the kernel center.
        delta: optional value added to the filtered pixels before storing them in dst.
        border_type: pixel extrapolation method.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the filtering.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04)
    """
    dst = _copy_if_not_inplace(x, inplace)

    return cv2.filter2D(
        x,
        dst,
        dddepth,
        kernel,
        anchor=anchor,
        delta=delta,
        borderType=BORDER_TYPES_DICT[border_type],
    )


def laplacian(
    x: np.ndarray,
    ddepth: int,
    ksize: int = 1,
    scale: float = 1.0,
    delta: float = 0.0,
    border_type: BorderTypes = "reflect101",
    *,
    inplace: bool = False,
) -> np.ndarray:
    r""" Calculates the Laplacian of an image.

    The function calculates the Laplacian of the source image by adding up the second x and y derivatives
    calculated using the Sobel operator:

    .. math::
        \texttt{dst} (x,y) =
        \Delta \texttt{src} (x,y) =
        \frac{d^2 \texttt{src} (x,y)}{dx^2} + \frac{d^2 \texttt{src} (x,y)}{dy^2}

    This is done when :math:`ksize > 1` . When ksize=1, the Laplacian is computed by filtering the image
    with the following :math:`3 \times 3` mask:

    .. math::
        \begin{bmatrix}
            0 & 1 & 0 \\\
            1 & -4 & 1 \\\
            0 & 1 & 0
        \end{bmatrix}

    Args:
        x: input image.
        ddepth: desired depth of the destination image.
        ksize: aperture size used to compute the second-derivative filters.
            See getDerivKernels for details. The size must be positive and odd.
        scale: optional scale factor for the computed Laplacian values.
        delta: optional delta value that is added to the results prior to storing them in dst.
        border_type: pixel extrapolation method.
        inplace: if True, the input image will be modified inplace.

    Returns:
        Result of the filtering.

    See Also:
        [OpenCV Doc](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6)
    """
    dst = _copy_if_not_inplace(x, inplace)

    return cv2.Laplacian(
        x,
        dst,
        ddepth,
        ksize=ksize,
        scale=scale,
        delta=delta,
        borderType=BORDER_TYPES_DICT[border_type],
    )
