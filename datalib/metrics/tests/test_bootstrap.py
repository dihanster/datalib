"""
Module with the test for the bootstraps wraping metrics functions.
"""

import pytest

from sklearn import datasets
from sklearn import metrics as sk_metrics

from .._bootstrap import bootstrap_metric

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_random_state


# iris
iris = datasets.load_iris()


@pytest.mark.parametrize(
    "metric, threshold_dependent, kwargs",
    [
        (sk_metrics.roc_auc_score, False, {}),
        (sk_metrics.average_precision_score, False, {}),
        (sk_metrics.accuracy_score, False, {}),
        (sk_metrics.f1_score, False, {}),
        (sk_metrics.fbeta_score, False, {"beta": 3}),
    ],
)
@pytest.mark.parametrize(
    "sample_weight",
    [None, check_random_state(42).exponential(size=iris.target.shape[0])])
def test_bootstrap_metric_classification(
    metric, threshold_dependent, kwargs, sample_weight
):
    X, y = iris.data, iris.target
    preprocessor = make_column_transformer(
        (StandardScaler(), [0, 2]), (RobustScaler(), [1, 3])
    )
    pipe = make_pipeline(preprocessor, LogisticRegression).fit(X, y)

    if threshold_dependent:
        pred = pipe.predict(X)
    else:
        pred = pipe.predict_proba(X)

    #TODO: Ok. it is running, but what do I want to assert here?
    bootstrap_metric(y, pred, metric, **kwargs)
