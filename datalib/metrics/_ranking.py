"""
Module containing the main metrics for ranking.
"""
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.utils import (
    assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    _check_pos_label_consistency,
)


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
    sample_weight = _check_sample_weight(
        sample_weight, y_true, only_non_negative=True
    )

    if y_type == "binary":
        ranking = np.argsort(y_score)[::-1]
        ranked = y_true[ranking] * sample_weight

        cumulative_gains = np.append(0, np.cumsum(ranked) / np.sum(ranked))
        thresholds = np.arange(0, len(ranked) + 1) / len(ranked)

        # TODO: Implement traditional gini calculation
        gini = (
            2 * roc_auc_score(y_true, y_score, sample_weight=sample_weight)
        ) - 1

        return cumulative_gains, thresholds, gini

    raise NotImplementedError("Only binary class supported!")


def delinquency_curve(y_true, y_score, pos_label=None):
    """Delinquency curve for a binary classification.

    The delinquency curve shows how the default rate (proportion of
    pos_labels) changes with different approval rates. The curve is
    typically plotted on a graph, with the approval rate on the x-axis
    and the default rate on the y-axis. The curve is created by sorting
    the samples by score and calculating the default rate for
    subsequently larger population getting the best scores at first.

    Deliquency curve is key on many actuarial operations, where grasping
    the relative percentage of misclassification on approval levels is
    vital.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1},
        then pos_label should be explicitly given.

    y_score : array-like, shape (n_samples,)
        Target scores, can either be probability estimates of the
        positive class, confidence values, or non-thresholded measure of
        decisions (as returned by "decision_function" on some
        classifiers).

    pos_label : int or str, default=None
        The label of the positive class.

    Returns
    -------
    approval_rate: array-like, shape (n_samples,).
        Increasing approval rate (percentage of approved best scores)
        used to compute `default_rate`. It lies in the support (0, 1).

    default_rate: array-like, shape (n_samples,).
        Default rates values for the approval rates such that the
        element i it proportion of delinquents when approving
        `approval_rate[i]` of the population.

    optimal_rate: array-like, shape (n_samples,).
        Optimal default rates for a perfect model.
    """
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided {labels}."
        )
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    check_consistent_length(y_true, y_score)
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    y_true = y_true == pos_label

    scores_idxs = np.argsort(y_score)[::1]
    actual_idxs = np.argsort(y_true)[::1]

    y_true_sorted_by_scores = y_true[scores_idxs].copy()
    y_true_sorted = y_true[actual_idxs].copy()

    list_index = np.arange(1, len(y_true_sorted_by_scores) + 1)
    approval_rate = np.append(0, list_index / len(list_index))
    default_rate = np.append(0, y_true_sorted_by_scores.cumsum() / list_index)
    optimal_rate = np.append(0, y_true_sorted.cumsum() / list_index)

    return approval_rate, default_rate, optimal_rate
