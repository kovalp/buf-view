import numpy as np

from .conftest import BufView


def test_exact(ebv: BufView[np.uint8]) -> None:
    assert ebv._get_new_size(345) == 345


def test_amortized_lt2x(abv: BufView[np.uint8]) -> None:
    assert abv._get_new_size(124) == 246


def test_amortized_gt2x(abv: BufView[np.uint8]) -> None:
    assert abv._get_new_size(1024) == 1024
