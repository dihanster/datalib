"""
Module with tests for the Delinquency display functions.
"""
import pytest

from ..delinquency_curve import DeliquencyDisplay
from ..._ranking import delinquency_curve
from numpy.testing import (
    assert_allclose,
)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def iris_data_binary(iris_data):
    X, y = iris_data
    return X[y < 2], y[y < 2]


def test_delinquency_display__assess_plot_parameters(iris_data_binary):
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
        == "Relative % of approvals on the population (Positive class: 1)"
    )
    assert viz.ax_.get_ylabel() == "Default Rate (Positive class: 1)"

    expected_legend_labels = ["LogisticRegression", "The optimal default rate"]
    legend_labels = viz.ax_.get_legend().get_texts()
    assert len(legend_labels) == len(expected_legend_labels)
