import numpy as np
import pytest

from .conftest import BufView


def test_none(ebv: BufView[np.uint8]) -> None:
    ebv._buf.fill(123)
    view = ebv._get_view(45, None)
    assert view == pytest.approx(123)
    assert view.shape == (45,)
    assert view.dtype == np.uint8
    assert np.shares_memory(view, ebv._buf)


def test_int(ebv: BufView[np.uint8]) -> None:
    view = ebv._get_view(45, 34)
    assert view == pytest.approx(34)
    assert view.shape == (45,)
    assert view.dtype == np.uint8
    assert np.shares_memory(view, ebv._buf)


def test_array(ebv: BufView[np.uint8]) -> None:
    data = np.linspace(1, 45, 45, dtype=np.uint8)
    view = ebv._get_view(45, data)
    assert view == pytest.approx(data)
    assert view.shape == (45,)
    assert view.dtype == np.uint8
    assert np.shares_memory(view, ebv._buf)
