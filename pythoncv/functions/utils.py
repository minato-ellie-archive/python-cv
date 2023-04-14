from typing import Tuple

import cv2  # type: ignore
import numpy as np

from pythoncv.core.types.dtype import CVDType


def get_deriv_kernels(dx: int = 1,
                      dy: int = 1,
                      ksize: int = 3,
                      normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """ Get the kernels for computing the derivatives.

    Args:
        dx:
            Order of the derivative x.
        dy:
            Order of the derivative y.
        ksize:
            Size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
        normalize:
            Flag indicating whether to normalize (scale down) the filter coefficients or not.
            Theoretically, norm should be true but practically that often leads to
            visibly sharper images, especially when the following filtering function is
            used to filter floating-point images (see filter2D() description), because
            of the possible loss of fractions due to arithmetics.
            When the function is used to filter 8-bit images, there is no such a problem.
            So, in the end, you may just set norm = normalize to simplify the call.

    Returns:
        The kernels for computing the derivatives.

    """
    kx = cv2.getDerivKernels(dx, dy, ksize, normalize=normalize)[0]
    ky = cv2.getDerivKernels(dy, dx, ksize, normalize=normalize)[0]
    return kx, ky


def get_gabor_kernel(
    ksize: Tuple[int, int],
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float,
    psi: float = 0,
    ktype: CVDType = 'float64',
) -> np.ndarray:
    """ Get the Gabor kernel.

    Args:
        ksize:
            Size of the filter returned.
        sigma:
            Standard deviation of the gaussian envelope.
        theta:
            Orientation of the normal to the parallel stripes of a Gabor function.
        lambd:
            Wavelength of the sinusoidal factor.
        gamma:
            Spatial aspect ratio.
        psi:
            Phase offset.
        ktype:
            Type of filter coefficients. It can be 'float32' or 'float64'.

    Returns:
        The Gabor kernel.

    """
    return cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)


def get_gaussian_kernel(ksize, sigma, ktype: CVDType = 'float64'):
    """ Get the Gaussian kernel.

    Args:
        ksize:
            Aperture size. It should be odd and positive.
        sigma:
            Gaussian standard deviation. If it is non-positive, it is computed from ksize as
            sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
        ktype:
            Type of filter coefficients. It can be 'float32' or 'float64'.

    Returns:
        The Gaussian kernel.

    """
    return cv2.getGaussianKernel(ksize, sigma, ktype)


def get_structuring_element(
        shape: int,
        ksize: Tuple[int, int],
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = 1,
) -> np.ndarray:
    """ Get the structuring element.

    Args:
        shape:
            Shape of the structuring element:
            - cv2.MORPH_RECT
            - cv2.MORPH_ELLIPSE
            - cv2.MORPH_CROSS
        ksize:
            Size of the structuring element.
        anchor:
            Anchor position within the element. The default value (-1, -1) means that the
            anchor is at the element center.
        iterations:
            Number of times erosion and dilation are applied.

    Returns:
        The structuring element.

    """
    return cv2.getStructuringElement(shape, ksize, anchor=anchor)
