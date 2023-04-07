""" Transformer is a function that converts a value(like CVImage) to another value.

For example, Filter is a function that converts a CVImage to another CVImage,
and new CVImage have similar shape with the original one.

In PythonCV we have three kinds of transformers:

    - **Filter**: convert a CVImage to another CVImage. New CVImage have similar shape with the original one.
    - **Converter**: convert a CVImage to another CVImage. New CVImage have different shape with the original one.
    - **Tool**: convert a CVImage to another value. New value can be any type.

> In other words, Filter is a special Converter, and Converter is a special Tool.
> You can always use Filter's result as Filter's input, whatever how many times you use it.
>
> Most of the time, you can use Converter's result as Converter's input, but when shape of the result
> is too small or too large, you can't use it as some Converter's input.

If you want to write your own transformer, you can extend Transformer class, and implement __call__ method.
**But using Current Transformer Type is highly recommended**, so user can know what kind of transformer it is.

"""

from .common import Pipeline

__all__ = ['Pipeline', 'filters', 'converters', 'tools', 'common']
