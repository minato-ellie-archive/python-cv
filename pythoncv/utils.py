""" Utility functions for pythoncv.

This module contains utility functions for pythoncv.(e.g. decorator, etc.)
Most time, you will not need to use these functions directly.

"""
import functools as ft
import inspect
import warnings
from typing import _Final  # type: ignore
from typing import Union


# TODO: Add support for type hints, and type checking fro typing.
def type_assert(**type_assertions):
    """ Decorator to assert the type of function arguments.

    Args:
        **type_assertions:
            Keyword arguments where the key is the name of the function argument,
            and the value is the type that the argument should be.

    Examples:
        >>> @type_assert(a=int, b=int)
        ... def foo(a, b):
        ...     return a + b
        >>> foo(1, '2')
        Traceback (most recent call last):
        ...
        TypeError: Argument 'b' must be of type '<class 'int'>'
        >>> foo(1, 2)
        3

        >>> @type_assert(a=int, b=(int, float))
        ... def foo(a, b):
        ...     return a + b
        >>> foo(1, 2)
        3
        >>> foo(1, 2.0)
        3.0
        >>> foo(1, '2')
        Traceback (most recent call last):
        ...
        TypeError: Argument 'b' must be of type '(<class 'int'>, <class 'float'>)'

        >>> @type_assert()
        ... def foo(a: int, b: float) -> float:
        ...     return float(a + b)
        >>> foo(1, 2)
        3.0
        >>> foo(1, '2')
        Traceback (most recent call last):
        ...
        TypeError: Argument 'b' must be of type '<class 'float'>'

        >>> @type_assert(b=(int, float)) # type annotations is necessary, when using Type in typing
        ... def foo(a: int, b: Union[int, float]) -> float:
        ...     return float(a + b)
        >>> foo(1, 2)
        3.0
        >>> foo(1, 2.0)
        3.0
        >>> foo(1, '2')
        Traceback (most recent call last):
        ...
        TypeError: Argument 'b' must be of type '(<class 'int'>, <class 'float'>)'

    Returns:
        The decorated function.

    Notes:
        This decorator is only active when the python interpreter is run with the -O flag.

    Warnings:
        `type_assert()` will not work, when using `Type` in `typing` package.
    """

    def decorator(func):
        if not __debug__:
            return func

        # map function argument names to supplied type assertions
        annotations = func.__annotations__
        annotations.update(type_assertions)

        block_list = [key for (key, value) in annotations.items() if isinstance(value, _Final)]
        for key in block_list:
            if key not in type_assertions:
                warnings.warn(
                    f"Type assertion must be set for '{key}' argument, when using Type in typing package.",
                    SyntaxWarning,
                )
                del annotations[key]

        # get the signature of the function
        sig = inspect.signature(func)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for name, value in bound_args.arguments.items():
                if name in annotations:
                    if not isinstance(value, annotations[name]):
                        raise TypeError(f"Argument '{name}' must be of type '{annotations[name]}'")

            return func(*args, **kwargs)

        return wrapper

    return decorator
