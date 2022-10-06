from ..metrics import delinquency_curve, DeliquencyDisplay
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import numpy as np
import pytest


def test_delinquency_curve__success_case():

    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8])
    approval_rate, default_rate, optimal_rate = delinquency_curve(y_true, y_scores)

    assert_array_equal(approval_rate, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    assert_almost_equal(default_rate, np.array([0, 0, 0.5, 0.3333333, 0.5]), decimal=5)
    assert_almost_equal(optimal_rate, np.array([0, 0, 0, 0.3333333, 0.5]), decimal=5)


def test_delinquency_curve__multilabel_excpetion():

    y_true = np.array([0, 0, 1, 1, 2])
    y_scores = np.array([0.1, 0.4, 0.3, 0.8, 0.04])
    with pytest.raises(Exception) as exc_info:
        _, _ = delinquency_curve(y_true, y_scores)
    assert (
        str(exc_info.value)
        == "Only binary classification is supported. Provided labels [0 1 2]."
    )


@pytest.fixture(scope="module")
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def iris_data_binary(iris_data):
    X, y = iris_data
    return X[y < 2], y[y < 2]


def test_calibration_display_compute(iris_data_binary):
    X, y = iris_data_binary

    lr = LogisticRegression().fit(X, y)

    viz = DeliquencyDisplay.from_estimator(lr, X, y)

    y_prob = lr.predict_proba(X)[:, 1]
    approval_rate, default_rate, optimal_rate = delinquency_curve(y, y_prob)

    assert_allclose(viz.approval_rate, approval_rate)
    assert_allclose(viz.default_rate, default_rate)
    assert_allclose(viz.optimal_rate, optimal_rate)

    assert viz.estimator_name == "LogisticRegression"

    # cannot fail thanks to pyplot fixture
    import matplotlib as mpl  # noqa

    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)

    assert (
        viz.ax_.get_xlabel()
        == "Relative percentage of approvals on the population (Positive class: 1)"
    )
    assert viz.ax_.get_ylabel() == "Default Rate (Positive class: 1)"

    expected_legend_labels = ["LogisticRegression", "The optimal default rate"]
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels
