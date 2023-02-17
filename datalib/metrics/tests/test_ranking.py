import numpy as np
import pytest

from .._ranking import delinquency_curve
from numpy.testing import assert_array_equal, assert_almost_equal
from sklearn.datasets import load_iris


@pytest.fixture(scope="module")
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def iris_data_binary(iris_data):
    X, y = iris_data
    return X[y < 2], y[y < 2]


def test_delinquency_curve__initial_and_end_values():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    approval_rate, _, _ = delinquency_curve(y_true, y_scores)

    assert approval_rate[0] == 0
    assert approval_rate[-1] == 1


def test_delinquency_curve__success_case():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    approval_rate, default_rate, optimal_rate = delinquency_curve(
        y_true, y_scores
    )

    assert_array_equal(approval_rate, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    assert_almost_equal(
        default_rate, np.array([0, 0, 0.5, 0.3333333, 0.5]), decimal=5
    )
    assert_almost_equal(
        optimal_rate, np.array([0, 0, 0, 0.3333333, 0.5]), decimal=5
    )


def test_delinquency_curve__multilabel_exception():
    y_true = np.array([0, 0, 1, 1, 2])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8, 0.04])
    with pytest.raises(Exception) as exc_info:
        _, _ = delinquency_curve(y_true, y_scores)
    assert (
        str(exc_info.value)
        == "Only binary classification is supported. Provided [0 1 2]."
    )
