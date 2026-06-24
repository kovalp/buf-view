"""."""

import numpy as np

from .conftest import BufView


def test_init(ebv: BufView[np.uint8]) -> None:
    assert ebv._buf.dtype == np.uint8
    assert ebv._buf.shape == (123,)
    assert ebv.exact


def test_amortized(abv: BufView[np.uint8]) -> None:
    assert abv._buf.dtype == np.uint8
    assert abv._buf.shape == (123,)
    assert not abv.exact
