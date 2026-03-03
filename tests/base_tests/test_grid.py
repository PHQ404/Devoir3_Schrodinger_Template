import numpy as np
import pytest

from src.fem import Grid


@pytest.mark.parametrize(
    "points,expected",
    [
        (
            np.arange(5),
            np.array([[2 / 3, 1 / 6, 0], [1 / 6, 2 / 3, 1 / 6], [0, 1 / 6, 2 / 3]]),
        )
    ],
)
def test_inner_mass_matrix(points, expected):
    grille = Grid(points)
    np.testing.assert_allclose(grille.inner_mass_matrix().toarray(), expected)


@pytest.mark.parametrize(
    "points,expected", [(np.arange(5), np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]]))]
)
def test_inner_laplacian_matrix(points, expected):
    grille = Grid(points)
    np.testing.assert_allclose(grille.inner_laplacian_matrix().toarray(), expected)


@pytest.mark.parametrize(
    "points,potential_func,expected",
    [
        (
            np.arange(5),
            lambda x: x**2,
            np.array([[0.733, 0.383, 0], [0.383, 2.733, 1.05], [0, 1.05, 6.067]]),
        )
    ],
)
def test_potential_matrix(points, potential_func, expected):
    grille = Grid(points)
    np.testing.assert_allclose(
        grille.potential_matrix(potential_func).toarray(),
        expected,
        atol=1e-3,
    )
