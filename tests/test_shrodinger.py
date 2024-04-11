import numpy as np
import pytest
import pickle
import os
from src import fem
from src import shooting


def sqr_potential(x):
    return x * x/2


potential_funcs = {func.__name__: func for func in [sqr_potential]}


test_data = {}
for function in ["solve_shrodinger_using_fem", "solve_shrodinger_using_shooting"]:
    test_data[function] = pickle.load(
        open(os.path.join(os.path.dirname(__file__), "test_data", f"{function}.pkl"), "rb")
    )


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (inputs_expected_dict["inputs"], inputs_expected_dict["output"])
        for inputs_expected_dict in test_data["solve_shrodinger_using_fem"]
    ],
)
def test_solve_shrodinger_using_fem(inputs, expected):
    inputs = (potential_funcs[inputs[0]], *inputs[1:])
    eig_pred, eig_vect_pred = fem.solve_shrodinger_using_fem(*inputs)
    eig_target, eig_vect_target = expected
    eig_vect_pred, eig_vect_target = np.abs(eig_vect_pred)**2, np.abs(eig_vect_target)**2

    np.testing.assert_allclose(
        eig_pred, eig_target,
        atol=1e-2, rtol=1e-2,
        err_msg=f"Eigenvalues are not close enough. Expected: {eig_target}, got: {eig_pred}"
    )

    np.testing.assert_allclose(
        eig_vect_pred, eig_vect_target,
        atol=1e-2, rtol=1e-2,
        err_msg=f"Eigenvectors are not close enough. Expected: {eig_vect_target}, got: {eig_vect_pred}"
    )


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (inputs_expected_dict["inputs"], inputs_expected_dict["output"])
        for inputs_expected_dict in test_data["solve_shrodinger_using_shooting"]
    ],
)
def test_solve_shrodinger_using_shooting(inputs, expected):
    inputs = (potential_funcs[inputs[0]], *inputs[1:])
    eig_pred, eig_vect_pred = shooting.solve_shrodinger_using_shooting(*inputs)
    eig_target, eig_vect_target = expected
    eig_vect_pred, eig_vect_target = np.abs(eig_vect_pred)**2, np.abs(eig_vect_target)**2

    np.testing.assert_allclose(
        eig_pred, eig_target,
        atol=1e-2, rtol=1e-2,
        err_msg=f"Eigenvalues are not close enough. Expected: {eig_target}, got: {eig_pred}"
    )

    np.testing.assert_allclose(
        eig_vect_pred, eig_vect_target,
        atol=1e-2, rtol=1e-2,
        err_msg=f"Eigenvectors are not close enough. Expected: {eig_vect_target}, got: {eig_vect_pred}"
    )

