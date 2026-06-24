"""."""

import numpy as np
import pytest

from buf_view import BufView


@pytest.fixture
def ebv() -> BufView:
    return BufView(123, np.uint8, "exact")


@pytest.fixture
def abv() -> BufView:
    return BufView(123, np.uint8, "2x")
