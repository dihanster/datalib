"""
Module with the test for the bootstraps wraping metrics functions.
"""

import pytest

from sklearn import datasets
from sklearn import metrics as sk_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_random_state

from .._bootstrap import bootstrap_metric


# iris
iris = datasets.load_iris()


@pytest.mark.parametrize(
    "metric",
    [
        sk_metrics.accuracy_score,
        sk_metrics.f1_score,
        functools.partial(sk_metrics.fbeta_score, beta=3, average="macro"),
    ],
)
@pytest.mark.parametrize(
    "sample_weight",
    [check_random_state(42).exponential(size=iris.target.shape[0])],
)
def test_bootstrap_metric_binary_classification(metric, sample_weight):
    # TODO: non-threshold metrics, regression, multiclass...
    X, y = iris.data, (iris.target >= 1).astype(int)
    model = LogisticRegression(random_state=42).fit(X, y)
    pred = model.predict(X)

    # TODO: Ok. it is running, but what do I want to assert here?
    bootstrap_metric(
        metric,
        y,  # y_true will enter as a arg
        pred,  # y_pred will enter as a arg
        sample_weight=sample_weight,  # sample_weight will enter as a kwarg
    )
