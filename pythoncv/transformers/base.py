from functools import wraps
from typing import (Any, Callable, TypeVar)

import numpy as np

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class Transformer:

    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, x):
        return self._fn(x)


class Filter(Transformer):

    def __call__(self, x: np.ndarray) -> Callable[[
            np.ndarray,
    ], np.ndarray]:
        return super().__call__(x)

    @classmethod
    def make(cls, fn: Callable[..., np.ndarray], *args, **kwargs) -> 'Filter':

        @wraps(fn)
        def wrapper(x: np.ndarray) -> np.ndarray:
            return fn(x, *args, **kwargs)

        return cls(wrapper)


class Converter(Transformer):

    def __call__(self, x: np.ndarray) -> Callable[[
            np.ndarray,
    ], np.ndarray]:
        return super().__call__(x)

    @classmethod
    def make(cls, fn: Callable[..., np.ndarray], *args, **kwargs) -> 'Converter':

        @wraps(fn)
        def wrapper(x: np.ndarray) -> np.ndarray:
            return fn(x, *args, **kwargs)

        return cls(wrapper)


class Tool(Transformer):

    def __call__(self, x: np.ndarray) -> Callable[[
            np.ndarray,
    ], Any]:
        return super().__call__(x)

    @classmethod
    def make(cls, fn: Callable[..., np.ndarray], *args, **kwargs) -> 'Tool':

        @wraps(fn)
        def wrapper(x: np.ndarray) -> np.ndarray:
            return fn(x, *args, **kwargs)

        return cls(wrapper)
