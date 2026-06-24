import math
from typing import Generic, Literal, TypeVar

import numpy as np

TT3 = tuple[int, int, int]
TT4 = tuple[int, int, int, int]
TT5 = tuple[int, int, int, int, int]
ShapeT = TypeVar("ShapeT", tuple[int], tuple[int, int], TT3, TT4, TT5)
T = TypeVar("T", bound=np.generic)
GT = Literal["exact", "2x"]
CT = np.ndarray[ShapeT, np.dtype[T]]
ODT = np.ndarray[tuple[int], np.dtype[T]]


class BufView(Generic[T]):
    def __init__(self, ini_size: int, dtype: type[T], growth: GT) -> None:
        self.dtype = dtype
        self._buf: ODT = np.zeros(ini_size, dtype=dtype)
        self.exact = growth == "exact"

    def _get_new_size(self, size: int) -> int:
        return size if self.exact else max(size, self._buf.size * 2)

    def _get_view(self, size: int, content: CT | None | float) -> ODT:
        view = self._buf[:size]
        if isinstance(content, int | float):
            view.fill(content)
        elif isinstance(content, np.ndarray):
            view[:] = content.reshape(size)
        return view

    def get_it(self, shape: ShapeT, content: CT | None | float) -> CT:
        size = int(math.prod(shape))
        if size > self._buf.size:
            self._buf = np.zeros(self._get_new_size(size), dtype=self.dtype)
        return self._get_view(size, content).reshape(shape)
