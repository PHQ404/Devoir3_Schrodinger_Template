"""Collections of helper tools for solving the Schrödinger equation."""

import time
from collections.abc import Callable
from functools import wraps

import numpy as np


def time_it(func: Callable) -> Callable:
    """Wrapper that prints the processing time of a given function."""

    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f}.")

        return result

    return time_it_wrapper


def lin_extrapolate(xs: np.ndarray, ys: np.ndarray, xf: float) -> float:
    """Linearly extrapolates a point given 2 reference points.

    :param xs: The x positions of the 2 reference points.
    :type xs: np.ndarray
    :param ys: The y positions of the 2 reference points.
    :type ys: np.ndarray
    :param xf: The x position of the point to extrapolate.
    :type xf: float
    :return: The extrapolated y value.
    :rtype: float
    """
    a = (ys[0] - ys[1]) / (xs[0] - xs[1])
    return ys[0] + a * (xf - xs[0])


def normalise(state: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Normalises a state vector assuming the vector is real.

    :param state: The state vector to normalise.
    :type state: np.ndarray
    :param xs: The x positions of the grid points of shape (N).
    :type xs: np.ndarray
    :return: The normalised state vector.
    :rtype: np.ndarray
    """
    dxs = xs[1:] - xs[:-1]
    temp = state * state
    area = 1 / 2 * (temp[:-1] + temp[1:]) * dxs
    area = area.sum()
    return state / np.sqrt(area)


def scope_roots(
    func: Callable, x0: float, args: tuple = (), max_iters: int = 500
) -> tuple[float, bool]:
    """Scope roots around point x=x0 using iterations of delta=1e-3.

    :param func: The function to scope roots for.
    :type func: callable
    :param x0: The point around which to scope roots.
    :type x0: float
    :param args: The arguments to pass to func.
    :type args: tuple
    :param max_iters: The maximum number of iterations to perform.
    :type max_iters: int
    :return: The root and whether the root was found.
    :rtype: tuple
    """
    raise NotImplementedError("This function is not implemented yet.")
