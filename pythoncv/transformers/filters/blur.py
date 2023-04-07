from typing import Tuple, Union

import pythoncv.functions as f
from pythoncv.transformers import Filter
from pythoncv.types.filter import BORDER_TYPES_DICT, BorderTypes
from pythoncv.utils import type_assert


@type_assert(
    ksize=(tuple, list, int),
    anchor=tuple,
    border_type=str,
)
def box_blur(
        ksize: Union[Tuple[int, int], int] = (3, 3),
        anchor: Tuple[int, int] = (-1, -1),
        normalize: bool = True,
        border_type: BorderTypes = "reflect101",
) -> Filter:
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    if len(ksize) != 2:
        raise ValueError(f"Invalid ksize: {ksize}")

    if border_type not in BORDER_TYPES_DICT:
        raise ValueError(f"Invalid border type: {border_type}")

    return Filter.make(f.box_blur, ksize, anchor, normalize, border_type)


@type_assert(
    ksize=(tuple, list),
    anchor=tuple,
    border_type=str,
)
def blur(ksize: Tuple[int, int] = (3, 3),
         anchor: Tuple[int, int] = (-1, -1), border_type: BorderTypes = "reflect101") -> Filter:
    if border_type not in BORDER_TYPES_DICT:
        raise ValueError(f"Invalid border type: {border_type}")

    return Filter.make(f.blur, ksize, anchor, border_type)


@type_assert(
    ksize=(tuple, list),
    sigma_x=(int, float),
    sigma_y=(int, float),
    border_type=str,
)
def gaussian_blur(ksize: Tuple[int, int] = (3, 3),
                  sigma_x: float = 0,
                  sigma_y: float = 0,
                  border_type: BorderTypes = "reflect101") -> Filter:
    if border_type not in BORDER_TYPES_DICT:
        raise ValueError(f"Invalid border type: {border_type}")

    return Filter.make(f.gaussian_blur, ksize, sigma_x, sigma_y, border_type)


@type_assert()
def median_blur(ksize: int = 3) -> Filter:
    return Filter.make(f.median_blur, ksize)


@type_assert(
    sigma_color=(int, float),
    sigma_space=(int, float),
)
def bilateral_filter(
    d: int = 5,
    sigma_color: float = 75,
    sigma_space: float = 75,
):
    return Filter.make(f.bilateral_filter, d, sigma_color, sigma_space)


@type_assert(radius=(tuple, list))
def stack_blur(radius: Tuple[int, int] = (3, 3),) -> Filter:
    return Filter.make(f.stack_blur, radius)


@type_assert(
    anchor=(tuple, list),
    border_type=str,
)
def square_blur(
        ksize: int = 3,
        anchor: Tuple[int, int] = (-1, -1),
        normalize: bool = True,
        border_type: BorderTypes = "reflect101",
) -> Filter:
    if border_type not in BORDER_TYPES_DICT:
        raise ValueError(f"Invalid border type: {border_type}")

    return Filter.make(f.square_blur, ksize, anchor, normalize, border_type)
