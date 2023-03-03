"""
Module containing the main metrics for ranking.
"""
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.utils import (
    check_array,
    check_consistent_length,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight


def cap_curve(
    y_true,
    y_score,
    sample_weight=None,
):
    """
    Base method to calculate the Cumulative Accuracy Profile Curve (CAP Curve).
    This metric ponders the rate of positive samples and the percentage of the
    dataset covered by each sequential cut-off threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.

    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a model / decision function.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    Returns
    -------
    cumulative_gains : ndarray of shape (n_samples,)
        Cumulative gain with each threshold (percentage of class 1).

    thresholds : ndarray of shape (n_samples,)
        Increasing thresholds (percentage of examples) on the decision
        function used to compute cap curve.

    gini: ndarray of shape (n_samples,)
        The normalized gini coefficient, calculated from the AUC.
    """
    y_type = type_of_target(y_true, input_name="y_true")
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)
    sample_weight = _check_sample_weight(sample_weight, y_true, only_non_negative=True)

    if y_type == "binary":
        ranking = np.argsort(y_score)[::-1]
        ranked = y_true[ranking] * sample_weight

        cumulative_gains = np.append(0, np.cumsum(ranked) / np.sum(ranked))
        thresholds = np.arange(0, len(ranked) + 1) / len(ranked)

        # TODO: Implement traditional gini calculation
        gini = (2 * roc_auc_score(y_true, y_score, sample_weight=sample_weight)) - 1

        return cumulative_gains, thresholds, gini

    raise NotImplementedError("Only binary class supported!")
