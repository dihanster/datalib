"""
Module with tests for the CAP Curve display functions.
"""
import matplotlib as mpl
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from datalib.metrics import cap_curve
from datalib.metrics import CAPCurveDisplay


@pytest.fixture(scope="module")
def iris_data():
    """A function to load the Iris dataset.
    """
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def iris_data_binary(iris_data):
    """A function to load the Iris dataset in a binary version.
    """
    X, y = iris_data
    return X[y < 2], y[y < 2]


def test_cap_display_compute(iris_data_binary):
    """Function to test the display/plot attributes for the CAP Curve.
    Check values to be displayed, titles and subtitles.
    """
    X, y = iris_data_binary

    clf = LogisticRegression().fit(X, y)

    viz = CAPCurveDisplay.from_estimator(clf, X, y)

    y_prob = clf.predict_proba(X)[:, 1]
    cumulative_gains, thresholds, gini = cap_curve(y, y_prob)

    assert_allclose(viz.cumulative_gains, cumulative_gains)
    assert_allclose(viz.thresholds, thresholds)
    assert_allclose(viz.gini, gini)

    assert viz.estimator_name == "LogisticRegression"

    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)

    assert viz.ax_.get_xlabel() == "% of Observations (Positive label: 1)"
    assert (
        viz.ax_.get_ylabel()
        == "% of Positive Observations (Positive label: 1)"
    )

    expected_legend_labels = [f"LogisticRegression (GINI = {gini:0.2f})"]
    legend_labels = viz.ax_.get_legend().get_texts()

    assert len(legend_labels) == len(expected_legend_labels)
    for labels in legend_labels:
        assert labels.get_text() in expected_legend_labels


def test_cap_curve_display_plotting():
    """TODO: Implement this test, based on scikit-learn's test for ROC Curve.
    """
