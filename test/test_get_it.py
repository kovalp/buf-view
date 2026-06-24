"""."""

import numpy as np

from .conftest import BufView


def test_without_reallocation(ebv: BufView) -> None:
    id_before_call = id(ebv._buf)
    view = ebv.get_it((3, 2), None)
    assert view.shape == (3, 2)
    assert view.dtype == np.uint8
    assert np.shares_memory(view, ebv._buf)
    assert id(ebv._buf) == id_before_call


def test_reallocation(ebv: BufView) -> None:
    id_before_call = id(ebv._buf)
    view = ebv.get_it((34, 12, 2), None)
    assert view.shape == (34, 12, 2)
    assert view.dtype == np.uint8
    assert np.shares_memory(view, ebv._buf)
    assert id(ebv._buf) != id_before_call
    assert ebv._buf.dtype == np.uint8
    assert ebv._buf.shape == (34 * 12 * 2,)


def test_amortized_reallocation(abv: BufView) -> None:
    abv.get_it((6, 12, 2), None)
    assert abv._buf.shape == (246,)
