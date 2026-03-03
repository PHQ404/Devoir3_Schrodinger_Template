"""
Implementation of the main functions for the finite element method
in one dimension.

This implementation is based on chapter 12 of the course notes
by David Sénéchal.
"""

from collections.abc import Callable

import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sp

from src import tools


class Grid:
    """Grid of points for the finite element method.

    :param points: List of sorted points of the grid.
    :type points: numpy.ndarray
    """

    def __init__(self, points: np.ndarray):
        # Ensure that the points are sorted.
        if np.any((points[1:] - points[:-1]) < 0.0):
            raise ValueError("Points must be sorted.")
        self.points = points

    def __len__(self) -> int:
        """Number of points in the grid."""
        return len(self.points)

    def inner_mass_matrix(self) -> sp.csc_array:
        """Returns the mass matrix of the grid
        as a scipy.sparse.csc_matrix.

        The boundary elements are not computed.
        Thus, element (0, 0) of the returned matrix
        represents the value (1, 1) of the full matrix.
        """
        n = len(self)
        matrix = sp.dok_matrix((n - 2, n - 2))
        # Compute values on the main diagonal.
        for site in range(1, n - 1):
            matrix[site - 1, site - 1] = (
                self.points[site + 1] - self.points[site - 1]
            ) / 3.0
        # Compute values on the off-diagonals.
        for site in range(1, n - 2):
            value = (self.points[site + 1] - self.points[site]) / 6.0
            matrix[site - 1, site] = value
            matrix[site, site - 1] = value
        return matrix.tocsc()

    def inner_laplacian_matrix(self) -> sp.csc_array:
        """Returns the matrix of the differential operator (D^2) of the grid
        as a scipy.sparse.csc_matrix.

        The boundary elements are not computed.
        Thus, element (0, 0) of the returned matrix
        represents the value (1, 1) of the full matrix.
        """
        n = len(self)
        matrix = sp.dok_matrix((n - 2, n - 2))
        # Compute values on the main diagonal.
        for site in range(1, n - 1):
            matrix[site - 1, site - 1] = sum(
                1.0 / (self.points[i] - self.points[i + 1]) for i in [site - 1, site]
            )
        # Compute values on the off-diagonals.
        for site in range(1, n - 2):
            value = 1.0 / (self.points[site + 1] - self.points[site])
            matrix[site - 1, site] = value
            matrix[site, site - 1] = value
        return matrix.tocsc()

    def potential_matrix(self, potential: Callable[[float], float]) -> sp.csr_array:
        """Returns the potential matrix of the grid
        as a scipy.sparse.csc_matrix.

        The boundary elements are not computed.
        Thus, element (0, 0) of the returned matrix
        represents the value (1, 1) of the full matrix.

        :param potential: A function from float to float representing the potential to compute.
        :type potential: Callable[[float], float]
        """

        # Compute the ratios for the integrals.
        def slope_pos(x, site):
            num = x - self.points[site - 1]
            denom = self.points[site] - self.points[site - 1]
            return num / denom

        def slope_neg(x, site):
            num = self.points[site + 1] - x
            denom = self.points[site + 1] - self.points[site]
            return num / denom

        n = len(self)
        matrix = sp.dok_matrix((n - 2, n - 2))

        # Compute values on the main diagonal.
        for site in range(1, n - 1):
            value_pos = integrate.quad(
                lambda x: potential(x) * slope_pos(x, site) ** 2,
                self.points[site - 1],
                self.points[site],
            )[0]
            value_neg = integrate.quad(
                lambda x: potential(x) * slope_neg(x, site) ** 2,
                self.points[site],
                self.points[site + 1],
            )[0]
            matrix[site - 1, site - 1] = value_pos + value_neg

        # Compute values on the off-diagonals.
        for site in range(1, n - 2):
            value = integrate.quad(
                lambda x: potential(x) * slope_pos(x, site + 1) * slope_neg(x, site),
                self.points[site],
                self.points[site + 1],
            )[0]
            matrix[site - 1, site] = value
            matrix[site, site - 1] = value
        return matrix.tocsr()


@tools.time_it
def solve_shrodinger_using_fem(
    potential: Callable[[float], float], points: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Function that solves the time independant shrodinger equation
    for a given 1D potential and returns the first "n_states" eigenstates
    of the hamiltonian. It calculates the wavefunction at every x=points
    using the finite element method.

    :param potential: The function that returns the potential at every x position.
    :type potential: Callable[[np.ndarray], np.ndarray]
    :param points: The x position of the grid points of shape (N).
    :type points: np.ndarray
    :param n_states: Number of eigenstates to find.
    :type n_states: int
    :return: The first "n_states" energies of the eigenstates and the first "n_states" eigenstates
        calculated at every grid points of shape (n_states, N).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    raise NotImplementedError("You need to implement this function.")
