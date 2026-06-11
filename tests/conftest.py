import pytest
from kms.mesh import make_grid, make_icosphere, make_torus


@pytest.fixture
def small_grid():
    return make_grid(5, 5)


@pytest.fixture
def small_icosphere():
    return make_icosphere(subdivisions=1)


@pytest.fixture
def small_torus():
    return make_torus(R=1.0, r=0.4, n_major=8, n_minor=6)
