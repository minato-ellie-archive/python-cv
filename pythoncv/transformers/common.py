from typing import Generic, TypeVar, Callable, Any, Union, Tuple, overload

from pythoncv.core.transformer import Transformer, Filter, Converter, Tool

TTransformer = TypeVar('TTransformer', bound=Transformer)
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class Pipeline(Generic[TInput, TOutput], Transformer):
    """ A pipeline of transformers.

    Methods:
        __Call__: Apply the pipeline to an input.

    Examples:
        >>> from pythoncv import CVImage
        >>> from pythoncv.transformers.filters import GaussianBlur
        >>> from pythoncv.transformers.common import Pipeline
        >>> fn = Pipeline(GaussianBlur(3), GaussianBlur(5))
        >>> img = CVImage.from_file('test.jpg')
        >>> img.shape
        (100, 100, 3)
        >>> img = fn(img)
        >>> img.shape  # shape is not changed, because GaussianBlur is a filter.
        (100, 100, 3)

    """

    def __init__(self, *transformers: TTransformer):
        """ Initialize a pipeline.

        Args:
            *transformers:
                Transformers to be applied in the pipeline.
                Data will be passed from the first transformer to the last one.


        """
        self._transformers = transformers
        super().__init__(self._generate_fn())

    def _generate_fn(self) -> Callable[[TInput], TOutput]:

        def fn(x: TInput) -> TOutput:
            for transformer in self._transformers:
                x = transformer(x)
            return x  # type: ignore

        return fn
